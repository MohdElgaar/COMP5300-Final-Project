import argparse, os

model_names = {'base': 'roberta-base',
        'twitter': 'cardiffnlp/twitter-roberta-base',
        'longformer': 'allenai/longformer-base-4096',
        }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='.')
    parser.add_argument('--data', default='snli_balanced')
    parser.add_argument('--aim_exp', default='entropy-curr')
    parser.add_argument('--ckpt')
    parser.add_argument('--ckpt_dir', default='/checkpoints')
    parser.add_argument('--model_name', default='base')
    parser.add_argument('--num_labels', type=int, default=3)
    parser.add_argument('--ent_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--grad_accumulation', type=int, default=1)
    parser.add_argument('--val_freq', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--acc_classes', default='entropy_class')
    parser.add_argument('--diff_score')
    parser.add_argument('--diff_score_id', type=int)
    parser.add_argument('--max_length', type=int, default=160)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--curr', default='sl')
    parser.add_argument('--sl_lam', type=float, default=1)
    parser.add_argument('--dp_alpha', type=float, default=0.9)
    parser.add_argument('--balance_logits', action='store_true')
    parser.add_argument('--diff_permute', action='store_true')
    parser.add_argument('--sel_bp', action='store_true')
    parser.add_argument('--lr_decay', action='store_true')
    parser.add_argument('--burn_in', type=float, default=0)
    parser.add_argument('--burn_out', type=float, default=0)
    parser.add_argument('--ent_alpha', type=float, default=1)
    parser.add_argument('--sl_mode', default='avg')
    parser.add_argument('--spl_mode', default='easy')
    parser.add_argument('--ent_cfg', default='6')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--save_losses', action='store_true')
    parser.add_argument('--seed', default = '0')
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--data_fraction', type=float, default=1.0)
    parser.add_argument('--lng')
    args = parser.parse_args()
    args.seed = [int(x) for x in args.seed.split(',')]
    args.model_name = model_names.get(args.model_name, args.model_name)
    os.makedirs(args.ckpt_dir, exist_ok = True)
    assert args.burn_in + args.burn_out <= 1
    return args
