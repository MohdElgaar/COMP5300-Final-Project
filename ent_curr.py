import json
import torch
from torch import nn


class EntropyCurriculum(nn.Module):
    def __init__(self, cfg_idx, epochs, 
            avgloss = False,
            alpha = 0.5,
            decay = 0.9, percentile = 0.7,
            cfg = None,
            ent_classes = 3):
        super().__init__()

        self.avgloss = avgloss

        self.ent_classes = ent_classes

        if avgloss:
            self.bns = nn.ModuleList([nn.BatchNorm1d(1, affine = False)
                for i in range(ent_classes)])

        self.epochs = epochs
        if cfg:
            self.cfg = cfg
        elif ent_classes != 3:
            self.cfg = {i: {"c1": 50, "c2": i/ent_classes}
                    for i in range(ent_classes)}
        else:
            with open('cfg/c1c2_%s.json'%cfg_idx) as f:
                cfg = json.load(f)
                self.cfg = {int(k): v for k,v in cfg.items()}

    def forward(self, loss, training_progress, ent_class, writer = None):
        if self.avgloss:
            for i in range(self.ent_classes):
                sub_batch = loss[ent_class == i]
                if sub_batch.shape[0] > 1:
                    self.bns[i](sub_batch.view(-1,1))

            entlist = ent_class.tolist()
            means = torch.tensor([self.bns[c].running_mean for c in entlist]).to(loss.device)
            stds = torch.sqrt(torch.tensor([self.bns[c].running_var for c in entlist]))\
                    .to(loss.device)
            diff = loss - means
            dev = diff / stds
            if (torch.abs(dev) >= 2).any():
                dev = dev.double()
                shift = torch.where(torch.abs(dev) < 2, 0., dev)
                shift = torch.where(dev >= 2, 1., shift)
                shift = torch.where(dev >= 3, 2., shift)
                shift = torch.where(dev <= -2, -1., shift)
                shift = torch.where(dev <= -3, -2., shift)
                shift = shift.cpu()

                ent_class += shift.long()
                ent_class = torch.clamp(ent_class, 0, self.ent_classes-1)

            # counts = [max((dev[ent_class == i]).size(0), 1) for i in range(3)]
            # counts_up = [((dev[ent_class == i] >= 2).sum()/counts[i]).item()
            #         for i in range(3)]
            # counts_up[2] = 0
            # counts_down = [-((dev[ent_class == i] <= -2).sum()/counts[i]).item()
            #         for i in range(3)]
            # counts_down[0] = 0

            # for i, c in enumerate(['easy', 'med', 'hard']):
            #     writer.track(counts_up[i], name = 'moved',
            #             context = {'split': 'train' if self.training else 'val',
            #                 'direction': 'up',
            #                 'subset': c})
            #     writer.track(counts_down[i], name = 'moved',
            #             context = {'split': 'train' if self.training else 'val',
            #                 'direction': 'down',
            #                 'subset': c})

        entlist = ent_class.tolist()
        c1 = torch.tensor([self.cfg[c]['c1'] for c in entlist]).to(loss.device)
        c2 = torch.tensor([self.cfg[c]['c2'] for c in entlist]).to(loss.device)
        x = c1*(training_progress-c2)
        conf = torch.sigmoid(x)

        return conf
