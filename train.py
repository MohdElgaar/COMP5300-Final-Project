import torch
import aim
import os, argparse, joblib, shutil, math, json
import numpy as np
import torch.nn.functional as F
from torch import nn
from time import time
from datetime import datetime
from options import parse_args
from model import Model
from superloss import SuperLoss
from datasets import load_from_disk
from spl import SPL
from mentornet import MentorNet
from ent_curr import EntropyCurriculum
from diff_pred_weighting import DPWeighting
from torch.utils.data import DataLoader
from data import MyDataset, get_dataloader
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoTokenizer, AdamW, logging, AutoConfig
from transformers import get_linear_schedule_with_warmup
from scipy.stats import zscore, pearsonr
logging.set_verbosity_error()

mean = lambda l: sum(l)/len(l) if len(l) > 0 else 0.

sca_names = "W,S,VP,C,T,DC,CT,CP,CN,MLS,MLT,MLC,C-S,VP-T,C-T,DC-C,DC-T,T-S,\
CT-T,CP-T,CP-C,CN-T,CN-C".split(',')
lca_names = "wordtypes,swordtypes,lextypes,slextypes,wordtokens,swordtokens,\
lextokens,slextokens,ld,ls1,ls2,vs1,vs2,cvs1,ndw,ndwz,ndwerz,ndwesz,ttr,\
msttr,cttr,rttr,logttr,uber,lv,vv1,svv1,cvv1,vv2,nv,adjv,advv,modv".split(',')

def get_dataloaders(args, tokenizer):
    def tokenize(x):
        if 'sentence1' in x:
            return tokenizer(x['sentence1'], x['sentence2'],
                    truncation=True, max_length = args.max_length)
        elif 't' in x:
            return tokenizer(x['t'], 
                    truncation=True, max_length = args.max_length)

    def collate_fn(batch):
        max_len = max([len(x['input_ids']) for x in batch])
        for i in range(len(batch)):
            batch[i]['input_ids'] += [tokenizer.pad_token_id] \
                    * max(max_len - len(batch[i]['input_ids']), 0)
            batch[i]['attention_mask'] += [tokenizer.pad_token_id] \
                    * max(max_len - len(batch[i]['attention_mask']), 0)
        return {k: torch.tensor([x[k] for x in batch]) for k in batch[0].keys()}

    data_dir = os.path.join(args.data_dir, args.data)
    data = load_from_disk(data_dir)

    train_dataset = data['train']

    n = len(train_dataset)
    if args.data_fraction < 1:
        ids = np.random.choice(n, int(args.data_fraction*n), replace=False)
        train_dataset = train_dataset.select(ids)

    if args.noise > 0:
        noisy_ids = np.random.choice(n, int(args.noise*n), replace=False)
        noisy_labels = {idx: l for idx,l in zip(noisy_ids,
            np.random.permutation(train_dataset[noisy_ids]['label']))}
        def process(sample, idx):
            if idx in noisy_ids:
                sample['label'] = noisy_labels[idx]
            return sample
        train_dataset = train_dataset.map(process, with_indices = True)

    if args.diff_permute:
        diff_class = 'loss_class' if args.curr == 'loss' else 'entropy_class'
        diff = np.random.permutation(train_dataset[diff_class])
        def process(sample, idx):
            sample[diff_class] = diff[idx]
            return sample
        train_dataset = train_dataset.map(process, with_indices = True)

    if (args.diff_score is not None)\
            and ('lca' in args.diff_score or 'sca' in args.diff_score):
        diff = [x[args.diff_score_id] for x in train_dataset[args.diff_score]]
        thresholds = [np.percentile(diff, i/args.ent_classes*100)
                for i in range(args.ent_classes)]
        def assign_class(row):
            ent_class = 0
            for i in range(args.ent_classes - 1, -1, -1):
                if row[args.diff_score][args.diff_score_id] >= thresholds[i]:
                    ent_class = i
                    break
            return {'entropy_class': ent_class}
        
        train_dataset = train_dataset.map(assign_class)
        data['dev'] = data['dev'].map(assign_class)
        data['test'] = data['test'].map(assign_class)
    elif args.diff_score is not None and args.diff_score == 'anli_class':
        def assign_class(row):
            return {'entropy_class': row['anli_class']}
        train_dataset = train_dataset.map(assign_class, batched=True)
        data['dev'] = data['dev'].map(assign_class, batched=True)
        data['test'] = data['test'].map(assign_class, batched=True)
    elif args.ent_classes != 3:
        if args.curr == 'ent' or args.curr == 'ent+':
            thresholds = [np.percentile(data['train']['entropy'], i/args.ent_classes*100)
                    for i in range(args.ent_classes)]
            def assign_class(row):
                ent_class = 0
                for i in range(args.ent_classes - 1, -1, -1):
                    if row['entropy'] >= thresholds[i]:
                        ent_class = i
                        break
                return {'entropy_class': ent_class}

        elif args.curr == 'loss' or args.curr == 'loss+':
            thresholds = [np.percentile(data['train']['loss'], i/args.ent_classes*100)
                    for i in range(args.ent_classes)]
            def assign_class(row):
                loss_class = 0
                for i in range(args.ent_classes - 1, -1, -1):
                    if row['loss'] >= thresholds[i]:
                        loss_class = i
                        break
                return {'loss_class': loss_class}
        
        train_dataset = train_dataset.map(assign_class)
        data['dev'] = data['dev'].map(assign_class)
        data['test'] = data['test'].map(assign_class)

    dev_dataset = data['dev']
    test_dataset = data['test']
    columns = ['label']
    if 'diff' in data['train'].column_names:
        columns.append('diff')
    if 'entropy_class' in data['train'].column_names:
        columns.append('entropy_class')

    if 'snli' in data_dir or 'anli' in data_dir:
        text = ['sentence1', 'sentence2']
    else:
        text = ['t']
    columns += text

    if 'lng' in args.data:
        for t in text:
            columns.extend(['%s_lca'%t, '%s_sca'%t])

    if 'ins_weight' in data['train'].column_names:
        columns.append('ins_weight')

    if 'loss_class' in data['train'].column_names:
        columns.append('loss_class')

    if 'anli_class' in data['train'].column_names:
        columns.append('anli_class')

    train_dataset.set_format(type=None,
            columns=columns)
    dev_dataset.set_format(type=None,
            columns=columns)
    test_dataset.set_format(type=None,
            columns=columns)

    train_dataset = train_dataset.map(tokenize, batched = True, remove_columns = text)
    dev_dataset = dev_dataset.map(tokenize, batched = True, remove_columns = text)
    test_dataset = test_dataset.map(tokenize, batched = True, remove_columns = text)

    train_dataloader = DataLoader(train_dataset, args.batch_size, True, collate_fn = collate_fn, num_workers=0)
    dev_dataloader = DataLoader(dev_dataset, args.batch_size, collate_fn = collate_fn)
    test_dataloader = DataLoader(test_dataset, 1, collate_fn = collate_fn)

    return train_dataloader, dev_dataloader, test_dataloader,\
            train_dataset, dev_dataset


def init_model(args, device, ent_cfg=None):
    if args.ckpt:
        print('[Resuming]')
        state = torch.load(args.ckpt)
        step = state['step']
        name = os.path.basename(args.ckpt)
        str_end = name.rfind('_', 0, -8)
        name = name[:str_end]
        model = Model(args).to(device)

        if args.curr == 'sl':
            curr = SuperLoss(mode=args.sl_mode, lam=args.sl_lam).to(device)
            # curr.load_state_dict(state['curr'])
        elif args.curr == 'ent' or args.curr == 'loss':
            curr = EntropyCurriculum(args.ent_cfg, args.epochs, cfg=ent_cfg, ent_classes = args.ent_classes)
        elif args.curr == 'ent+':
            curr = EntropyCurriculum(args.ent_cfg, args.epochs, avgloss = True, cfg=ent_cfg, ent_classes = args.ent_classes)
        elif args.curr == 'spl':
            curr = SPL(mode = args.spl_mode)
        elif args.curr == 'mentornet':
            curr = MentorNet(args.num_labels, args.epochs).to(device)
        elif args.curr == 'dp':
            curr = DPWeighting(args.dp_tao, args.dp_alpha)
        else:
            curr = None
    else:
        step = 0
        name = datetime.now().strftime('%b%d_%H-%M-%S')
        if args.curr == 'sl':
            name += '_sl_%s'%args.sl_mode 
        elif args.curr == 'ent' or args.curr == 'ent+':
            name += '_ent_%s'%args.ent_cfg
        elif args.curr == 'loss':
            name += '_loss_%s'%args.ent_cfg
        elif args.curr == 'spl':
            name += '_spl_%s'%args.spl_mode
        elif args.curr == 'mentornet':
            name += '_mentornet'
        elif args.curr == 'dp':
            name += '_dp'
        model = Model(args).to(device)

        if args.curr == 'sl':
            curr = SuperLoss(mode=args.sl_mode, lam=args.sl_lam).to(device)
        elif args.curr == 'ent' or args.curr == 'loss':
            curr = EntropyCurriculum(args.ent_cfg, args.epochs, cfg=ent_cfg, ent_classes = args.ent_classes)
        elif args.curr == 'ent+' or args.curr == 'loss+':
            curr = EntropyCurriculum(args.ent_cfg, args.epochs, avgloss = True, cfg=ent_cfg, ent_classes = args.ent_classes)
        elif args.curr == 'spl':
            curr = SPL(mode = args.spl_mode)
        elif args.curr == 'mentornet':
            curr = MentorNet(args.num_labels, args.epochs).to(device)
        elif args.curr == 'dp':
            curr = DPWeighting(args.dp_tao, args.dp_alpha)
        else:
            curr = None

    if curr:
        curr.to(device)
    return model, curr, name, step

def init_opt(model, total_steps, args):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    if args.lr_decay:
        scheduler = get_linear_schedule_with_warmup(optimizer, -1, total_steps)
    else:
        scheduler = None

    if args.ckpt:
        state = torch.load(args.ckpt)
        optimizer.load_state_dict(state['optimizer'])
    return optimizer, scheduler

class Trainer():
    def __init__(self, model, tokenizer, crit, optimizer, scheduler, curr, epochs,
            writer, name, step, epoch_size, debug, device, args):
        self.model = model
        self.tokenizer = tokenizer
        self.crit = crit
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.curr = curr
        self.writer = writer
        self.name = name
        self.step = step
        self.epoch_size = epoch_size
        self.debug = debug
        self.best_acc = 0
        self.best_step = None
        self.device = device
        self.args = args
        self.epochs = epochs
        self.total_steps = epochs * epoch_size
        self.save_losses = args.save_losses
        if self.save_losses:
            self.losses = {'train': [], 'dev': []}

        self.loss_bn = nn.BatchNorm1d(1, affine = False).to(device)

    def get_loss(self, batch):
        x = {'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device)}

        logits = self.model(x)
        labels = batch['label'].to(self.device)
        loss_unw = self.crit(logits, labels)

        training_progress = self.step/self.total_steps
        relative_progress = (training_progress-self.args.burn_in)\
                /(1-self.args.burn_in-self.args.burn_out)

        if relative_progress >= 0\
                and relative_progress < 1\
                and self.curr is not None:
            if self.args.curr == 'mentornet':
                confs = self.curr(loss_unw, labels, self.step // self.epoch_size)
            elif self.args.curr == 'dp':
                confs = self.curr(loss_unw, batch['diff'].to(self.device))
            elif self.args.curr == 'loss' or self.args.curr == 'loss+':
                confs = self.curr(loss_unw,
                        relative_progress,
                        batch['loss_class'],
                        self.writer)
            else:
                confs = self.curr(loss_unw,
                        relative_progress,
                        batch['entropy_class'])
        else:
            confs = torch.ones_like(loss_unw)

        if 'ins_weight' in batch:
            weights = batch['ins_weight'].to(self.device)
        else:
            weights = torch.ones_like(loss_unw)

        confs = confs.reshape(-1)
        eps = 1e-5
        if self.args.balance_logits:
            loss_w = confs * weights * loss_unw
            total_loss = loss_w.sum() / max(eps, (confs*weights).sum())
        else:
            loss_w = (confs * loss_unw)
            if args.sel_bp:
                total_loss = loss_w[confs > 1e-5].sum() / max(eps, confs[confs > 1e-5].sum())
            else:
                total_loss = loss_w.sum() / max(eps, confs.sum())

        return logits, confs, total_loss, loss_unw.mean(), loss_unw, loss_w

    def evaluate(self, dataloader, count=None, return_pred = False, return_loss = False):
        losses = []
        losses_unw = []
        accs = []
        self.model.eval()
        if self.curr:
            self.curr.eval()
        acc_class = [[] for i in range(3)]
        loss_unw_class = [[] for i in range(3)]
        loss_w_class = [[] for i in range(3)]
        confs = [[] for i in range(3)]
        trues, preds = [], []
        full_loss = []
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                logits, conf, loss, loss_unw, all_loss_unw, all_loss_w = self.get_loss(batch)
            true = batch['label']
            pred = logits.argmax(-1).cpu()
            if return_loss:
                full_loss += all_loss_unw.tolist()
            acc = accuracy_score(true, pred)
            batch_accs = true == pred
            trues.extend(true.tolist())
            preds.extend(pred.tolist())
            losses_unw.append(loss_unw.detach().item())
            losses.append(loss.detach().item())
            accs.append(acc)
            for c in range(3):
                class_ids = batch[self.args.acc_classes] == c
                class_accs = batch_accs[class_ids]
                acc_class[c].extend(class_accs.tolist())

                class_loss_unw = all_loss_unw[class_ids]
                loss_unw_class[c].extend(class_loss_unw.tolist())

                class_loss_w = all_loss_w[class_ids]
                loss_w_class[c].extend(class_loss_w.tolist())
            if count and i > 0 and i % count == 0:
                break
            conf_easy = conf[batch[self.args.acc_classes] == 0]
            conf_med = conf[batch[self.args.acc_classes] == 1]
            conf_hard = conf[batch[self.args.acc_classes] == 2]
            if conf_easy.numel() != 0:
                confs[0].append(conf_easy.mean().item())
            if conf_med.numel() != 0:
                confs[1].append(conf_med.mean().item())
            if conf_hard.numel() != 0:
                confs[2].append(conf_hard.mean().item())

        self.model.train()
        if self.curr:
            self.curr.train()

        f1 = f1_score(trues, preds, average = 'macro')
        res = [mean(losses_unw), mean(losses), mean(accs),
                confs, f1,
                [mean(a) for a in acc_class],
                [mean(l) for l in loss_w_class],
                [mean(l) for l in loss_unw_class]
                ]

        if return_pred:
            res.append(preds)
        if return_loss:
            res.append(full_loss)

        return res

    def save(self):
        with open('{}/{}_args'.format(self.args.ckpt_dir, self.name), 'w') as f:
            json.dump(self.args.__dict__, f)
        self.model.backbone.save_pretrained('{}/{}_best_model'.format(self.args.ckpt_dir,
            self.name))
        torch.save({
            'model': {k: v for k,v in self.model.state_dict().items() if 'backbone' not in k},
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'curr': self.curr,
            'best_step': self.best_step,
            'best_acc': self.best_acc},
            '{}/{}_best_meta.pt'.format(self.args.ckpt_dir, self.name))

    def load(self, name):
        state = torch.load("%s_meta.pt"%name)
        self.model.load_state_dict(state['model'], strict=False)
        self.step = state['step']
        self.curr = state['curr']
        self.optimizer.load_state_dict(state['optimizer'])
        self.best_step = state['best_step']
        self.best_acc = state['best_acc']

        self.model.backbone = AutoModel.from_pretrained("%s_model"%name).to(self.device)

    def load_best(self):
        if self.best_step:
            print("[Loading Best] Current: %d -> Best: %d (%.4f)"%(self.step, self.best_step,
                self.best_acc))
        self.load('{}/{}_best'.format(self.args.ckpt_dir, self.name))
        return self.best_acc, self.best_step

    def cleanup(self):
        path = '{}/{}_best'.format(self.args.ckpt_dir, self.name)
        os.remove("%s_meta.pt"%path)
        shutil.rmtree("%s_model"%path)

    def train(self, train_dataloader, dev_dataloader, dev_dataset, train_ns = None):
        for e in range(self.epochs):
            self.model.train()
            for batch in train_dataloader:
                logits, conf, loss, loss_unw, all_loss_w, all_loss_unw = self.get_loss(batch)

                if args.grad_accumulation > 1:
                    loss /= args.grad_accumulation

                loss.backward()

                if (self.step+1) % args.grad_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler is not None:
                        self.scheduler.step()

                pred = logits.argmax(-1).cpu()
                true = batch['label']
                batch_accs = true == pred
                acc = accuracy_score(true, pred.cpu())
                f1 = f1_score(true, logits.argmax(-1).cpu(), average = 'macro')


                if self.step % 50 == 0 and self.writer is not None:
                    self.writer.track(loss.detach().item(), step = self.step,
                            epoch = self.step // self.epoch_size,
                            name = 'loss_weighted', context = {'split': 'train'})
                    self.writer.track(loss_unw.detach().item(), step = self.step,
                            epoch = self.step // self.epoch_size,
                            name = 'loss_unweighted', context = {'split': 'train'})
                    self.writer.track(acc, step = self.step,
                            epoch = self.step // self.epoch_size,
                            name = 'acc', context = {'split': 'train'})
                    for c,c_name in enumerate(['easy', 'med', 'hard']):
                        class_ids = batch[self.args.acc_classes] == c
                        class_conf = conf[class_ids]
                        class_acc = batch_accs[class_ids]
                        class_loss_w = all_loss_w[class_ids]
                        class_loss_unw = all_loss_unw[class_ids]
                        if class_conf.numel() != 0:
                            self.writer.track(class_conf.mean().item(), step = self.step,
                                epoch = self.step // self.epoch_size,
                                    name = 'conf', context = {'split': 'train', 'subset': c_name})
                            self.writer.track(mean(class_acc), step = self.step,
                                epoch = self.step // self.epoch_size,
                                    name = 'acc_bd',
                                    context = {'split': 'train', 'subset': c_name})
                            self.writer.track(class_loss_w.mean().item(), step = self.step,
                                epoch = self.step // self.epoch_size,
                                    name = 'loss_w_bd',
                                    context = {'split': 'train', 'subset': c_name})
                            self.writer.track(class_loss_unw.mean().item(), step = self.step,
                                epoch = self.step // self.epoch_size,
                                    name = 'loss_unw_bd',
                                    context = {'split': 'train', 'subset': c_name})

                    if 'lng' in args.data:
                        losses = all_loss_unw.detach().cpu()
                        self.writer.track(pearsonr(losses, 
                            batch['entropy_class'])[0],
                            step = self.step, epoch = self.step // self.epoch_size,
                            name = 'corr', context = {'split': 'train', 'corr': 'loss-entropy'})

                        if 'sentence1_lca' in batch:
                            for idx in range(len(lca_names)):
                                # if batch['lca'][idx].var() != 0:
                                r = pearsonr(losses,
                                        batch['sentence1_lca'][:,idx])[0]
                                self.writer.track(r,
                                    step = self.step, epoch = self.step // self.epoch_size,
                                    name = 'corr',
                                    context = {'split': 'train', 'corr': 'lca-%s'%lca_names[idx],
                                        'text': 'sentence1'})

                            for idx in range(len(lca_names)):
                                # if batch['lca'][idx].var() != 0:
                                r = pearsonr(losses,
                                        batch['sentence2_lca'][:,idx])[0]
                                self.writer.track(r,
                                    step = self.step, epoch = self.step // self.epoch_size,
                                    name = 'corr',
                                    context = {'split': 'train', 'corr': 'lca-%s'%lca_names[idx],
                                        'text': 'sentence2'})

                            for idx in range(len(sca_names)):
                                # if batch['sca'][idx].var() != 0:
                                    r = pearsonr(losses,
                                            batch['sentence1_sca'][:,idx])[0]
                                    self.writer.track(r,
                                        step = self.step, epoch = self.step // self.epoch_size,
                                        name = 'corr',
                                        context = {'split': 'train',
                                            'corr': 'sca-%s'%sca_names[idx], 'text': 'sentence1'})

                            for idx in range(len(sca_names)):
                                    r = pearsonr(losses,
                                            batch['sentence2_sca'][:,idx])[0]
                                    self.writer.track(r,
                                        step = self.step, epoch = self.step // self.epoch_size,
                                        name = 'corr',
                                        context = {'split': 'train',
                                            'corr': 'sca-%s'%sca_names[idx], 'text': 'sentence2'})
                        else:
                            for idx in range(len(lca_names)):
                                r = pearsonr(losses,
                                        batch['t_lca'][:,idx])[0]
                                self.writer.track(r,
                                    step = self.step, epoch = self.step // self.epoch_size,
                                    name = 'corr',
                                    context = {'split': 'train', 'corr': 'lca-%s'%lca_names[idx],
                                        'text': 't'})

                            for idx in range(len(sca_names)):
                                    r = pearsonr(losses,
                                            batch['t_sca'][:,idx])[0]
                                    self.writer.track(r,
                                        step = self.step, epoch = self.step // self.epoch_size,
                                        name = 'corr',
                                        context = {'split': 'train',
                                            'corr': 'sca-%s'%sca_names[idx], 'text': 't'})


                if (self.step + 1) % (self.epoch_size // self.args.val_freq) == 0:
                    res = self.evaluate(dev_dataloader, return_loss = True)
                    loss_unw, loss, acc, conf, f1,\
                            class_accs, class_loss_w, class_loss_unw = res[:8]
                    if self.save_losses:
                        res_train = self.evaluate(train_ns, return_loss = True)
                        self.losses['train'].append(res_train[-1])
                        self.losses['dev'].append(res[-1])

                    if self.writer is not None:
                        self.writer.track(loss_unw, name = 'loss_unweighted', step = self.step,
                            epoch = self.step // self.epoch_size,
                                context = {'split': 'val'})
                        self.writer.track(loss, name = 'loss_weighted', step = self.step,
                            epoch = self.step // self.epoch_size,
                                context = {'split': 'val'})
                        self.writer.track(acc, name = 'acc', step = self.step,
                            epoch = self.step // self.epoch_size,
                                context = {'split': 'val'})

                        self.writer.track(class_accs[0], step = self.step,
                            epoch = self.step // self.epoch_size,
                                name = 'acc_bd', context = {'split': 'val', 'subset': 'easy'})
                        self.writer.track(class_accs[1], step = self.step,
                            epoch = self.step // self.epoch_size,
                                name = 'acc_bd', context = {'split': 'val', 'subset': 'med'})
                        self.writer.track(class_accs[2], step = self.step,
                            epoch = self.step // self.epoch_size,
                                name = 'acc_bd', context = {'split': 'val', 'subset': 'hard'})

                        self.writer.track(class_loss_w[0], step = self.step,
                            epoch = self.step // self.epoch_size,
                                name = 'loss_w_bd', context = {'split': 'val', 'subset': 'easy'})
                        self.writer.track(class_loss_w[1], step = self.step,
                            epoch = self.step // self.epoch_size,
                                name = 'loss_w_bd', context = {'split': 'val', 'subset': 'med'})
                        self.writer.track(class_loss_w[2], step = self.step,
                            epoch = self.step // self.epoch_size,
                                name = 'loss_w_bd', context = {'split': 'val', 'subset': 'hard'})

                        self.writer.track(class_loss_unw[0], step = self.step,
                            epoch = self.step // self.epoch_size,
                                name = 'loss_unw_bd',
                                context = {'split': 'val', 'subset': 'easy'})
                        self.writer.track(class_loss_unw[1], step = self.step,
                            epoch = self.step // self.epoch_size,
                                name = 'loss_unw_bd',
                                context = {'split': 'val', 'subset': 'med'})
                        self.writer.track(class_loss_unw[2], step = self.step,
                            epoch = self.step // self.epoch_size,
                                name = 'loss_unw_bd',
                                context = {'split': 'val', 'subset': 'hard'})

                        self.writer.track(mean(conf[0]), step = self.step,
                            epoch = self.step // self.epoch_size,
                                name = 'conf', context = {'split': 'val', 'subset': 'easy'})
                        self.writer.track(mean(conf[1]), step = self.step,
                            epoch = self.step // self.epoch_size,
                                name = 'conf', context = {'split': 'val', 'subset': 'med'})
                        self.writer.track(mean(conf[2]), step = self.step,
                            epoch = self.step // self.epoch_size,
                                name = 'conf', context = {'split': 'val', 'subset': 'hard'})

                        # correlations
                        if 'lng' in args.data:
                            self.writer.track(pearsonr(res[-1], 
                                dev_dataset['entropy_class'])[0],
                                step = self.step, epoch = self.step // self.epoch_size,
                                name = 'corr', context = {'split': 'val', 'corr': 'loss-entropy'})

                            if 'sentence1_lca' in batch:
                                lca = np.array(dev_dataset['sentence1_lca'])
                                sca = np.array(dev_dataset['sentence1_sca'])
                                for idx in range(len(lca_names)):
                                    # if batch['lca'][idx].var() != 0:
                                    r = pearsonr(res[-1], 
                                           lca[:,idx])[0]
                                    self.writer.track(r,
                                        step = self.step, epoch = self.step // self.epoch_size,
                                        name = 'corr',
                                        context = {'split': 'val', 'corr': 'lca-%s'%lca_names[idx],
                                            'text': 'sentence1'})

                                for idx in range(len(sca_names)):
                                    r = pearsonr(res[-1], 
                                            sca[:,idx])[0]
                                    self.writer.track(r,
                                        step = self.step, epoch = self.step // self.epoch_size,
                                        name = 'corr',
                                        context = {'split': 'val', 'corr': 'sca-%s'%sca_names[idx],
                                            'text': 'sentence1'})

                                lca = np.array(dev_dataset['sentence2_lca'])
                                sca = np.array(dev_dataset['sentence2_sca'])
                                for idx in range(len(lca_names)):
                                    r = pearsonr(res[-1], 
                                           lca[:,idx])[0]
                                    self.writer.track(r,
                                        step = self.step, epoch = self.step // self.epoch_size,
                                        name = 'corr',
                                        context = {'split': 'val', 'corr': 'lca-%s'%lca_names[idx],
                                            'text': 'sentence2'})

                                for idx in range(len(sca_names)):
                                    r = pearsonr(res[-1], 
                                            sca[:,idx])[0]
                                    self.writer.track(r,
                                        step = self.step, epoch = self.step // self.epoch_size,
                                        name = 'corr',
                                        context = {'split': 'val', 'corr': 'sca-%s'%sca_names[idx],
                                            'text': 'sentence2'})
                            else:
                                lca = np.array(dev_dataset['t_lca'])
                                sca = np.array(dev_dataset['t_sca'])
                                for idx in range(len(lca_names)):
                                    # if batch['lca'][idx].var() != 0:
                                    r = pearsonr(res[-1], 
                                           lca[:,idx])[0]
                                    self.writer.track(r,
                                        step = self.step, epoch = self.step // self.epoch_size,
                                        name = 'corr',
                                        context = {'split': 'val', 'corr': 'lca-%s'%lca_names[idx],
                                            'text': 't'})

                                for idx in range(len(sca_names)):
                                    r = pearsonr(res[-1], 
                                            sca[:,idx])[0]
                                    self.writer.track(r,
                                        step = self.step, epoch = self.step // self.epoch_size,
                                        name = 'corr',
                                        context = {'split': 'val', 'corr': 'sca-%s'%sca_names[idx],
                                            'text': 't'})

                    if acc > self.best_acc:
                        self.best_acc = acc
                        self.best_step = self.step
                        self.save()

                self.step += 1

if __name__ == '__main__':
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataloader, dev_dataloader, test_dataloader,\
            train_dataset, dev_dataset = get_dataloaders(args, tokenizer)
    epoch_size = len(train_dataloader)
    total_steps = epoch_size * args.epochs

    if args.curr == 'dp':
        args.dp_tao = np.percentile(train_dataset['diff'], 50)

    for seed in args.seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        args.seed = seed

        model, curr, name, step = init_model(args, device)

        optimizer, scheduler = init_opt(model, total_steps, args)
        crit = nn.CrossEntropyLoss(reduction='none')

        writer = aim.Run(experiment=args.aim_exp, system_tracking_interval=None)\
                if not args.debug else None
        if writer is not None:
            writer['hparams'] = args.__dict__
            writer['meta'] = {'time': datetime.now().strftime('%y%m%d/%H:%M:%S')}

        trainer = Trainer(model, tokenizer, crit, optimizer, scheduler, curr, args.epochs,
                writer, name, step, epoch_size, args.debug, device, args)

        if not args.eval_only:
            print('[Starting Training]')
            trainer.train(train_dataloader, dev_dataloader,
                    dev_dataset,
                    DataLoader(train_dataset, args.batch_size) if args.save_losses else None)

        print('[Testing]')
        if args.save_losses:
            np.savez('losses/%s_%d.npz'%(args.data, seed), **trainer.losses)
        _, best_step = trainer.load_best()
        results = {}
        acc, confs, f1, class_acc  = trainer.evaluate(test_dataloader)[2:6]
        print('Acc:', acc)
        print('F1:', f1)
        print("0: {:.4f}\n1: {:.4f}\n2: {:.4f}".format(*class_acc))
        results['acc'] = acc
        results['f1'] = f1
        results['acc_easy'] = class_acc[0]
        results['acc_med'] = class_acc[1]
        results['acc_hard'] = class_acc[2]
        results['best_step'] = best_step
        if writer is not None:
            writer['results'] = results
        # trainer.cleanup()
