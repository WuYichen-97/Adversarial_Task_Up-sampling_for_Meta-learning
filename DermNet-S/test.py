import argparse
import random
import numpy as np
import torch
import os
from data_generator import MiniImagenet, ISIC, DermNet
from learner import Conv_Standard
# from protonet import Protonet
from maml import MAML

parser = argparse.ArgumentParser(description='MLTI')
parser.add_argument('--datasource', default='dermnet', type=str,
                    help='miniimagenet, isic, dermnet')
parser.add_argument('--num_classes', default=5, type=int,
                    help='number of classes used in classification (e.g. 5-way classification).')
parser.add_argument('--test_epoch', default=1000, type=int, help='test epoch, only work when test start')
parser.add_argument('--num_test_task', default=600, type=int, help='number of test tasks.')
## Training options
parser.add_argument('--metatrain_iterations', default=50000, type=int,
                    help='number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
parser.add_argument('--meta_batch_size', default=4, type=int, help='number of tasks sampled per meta-update')
parser.add_argument('--update_lr', default=0.01, type=float, help='inner learning rate')
parser.add_argument('--meta_lr', default=0.001, type=float, help='the base learning rate of the generator')
parser.add_argument('--num_updates', default=5, type=int, help='num_updates in maml')
parser.add_argument('--num_updates_test', default=10, type=int, help='num_updates in maml')
parser.add_argument('--update_batch_size', default=5, type=int,
                    help='number of examples used for inner gradient update (K for K-shot learning).')
parser.add_argument('--update_batch_size_eval', default=15, type=int,
                    help='number of examples used for inner gradient test (K for K-shot learning).')

## Model options
parser.add_argument('--num_filters', default=32, type=int,
                    help='number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')

## Logging, saving, and testing options
parser.add_argument('--logdir', default='xxx', type=str,
                    help='directory for summaries and checkpoints.')
parser.add_argument('--datadir', default='xxx', type=str, help='directory for datasets.')
parser.add_argument('--resume', default=0, type=int, help='resume training if there is a model available')
parser.add_argument('--train', default=0, type=int, help='True to train, False to test.')
parser.add_argument('--mix', default=0, type=int, help='use mixup or not')
parser.add_argument('--trial', default=0, type=int, help='trail for each layer')
parser.add_argument('--ratio', default=0.2, type=float, help='the ratio of meta-training tasks')


args = parser.parse_args()
print(args)

if args.datasource == 'isic':
    assert args.num_classes < 5

assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True

random.seed(1)
np.random.seed(2)

exp_string = 'ProtoNet_Cross' + '.data_' + str(args.datasource) + '.cls_' + str(args.num_classes) + '.mbs_' + str(
    args.meta_batch_size) + '.ubs_' + str(
    args.update_batch_size) + '.metalr' + str(args.meta_lr)

if args.num_filters != 64:
    exp_string += '.hidden' + str(args.num_filters)
if args.mix:
    exp_string += '.mix'
if args.trial > 0:
    exp_string += '.trial{}'.format(args.trial)
if args.ratio < 1.0:
    exp_string += '.ratio{}'.format(args.ratio)

print(exp_string)


def test(args, maml, test_epoch):
    res_acc = []
    args.meta_batch_size = 1
    
    random.seed(1)
    np.random.seed(2)
    if args.datasource == 'miniimagenet':
        dataloader = MiniImagenet(args, 'test')
    elif args.datasource == 'isic':
        dataloader = ISIC(args, 'test')
    elif args.datasource == 'dermnet':
        dataloader = DermNet(args, 'test')
    
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataloader):
        if step > args.num_test_task:
            break
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).cuda(), y_spt.squeeze(0).cuda(), \
                                     x_qry.squeeze(0).cuda(), y_qry.squeeze(0).cuda()
                                     
        _, acc_val = maml(x_spt, y_spt, x_qry, y_qry)
        res_acc.append(acc_val.item())

    res_acc = np.array(res_acc)
    print('test_epoch is {}, acc is {}, ci95 is {}'.format(test_epoch, np.mean(res_acc),
                                                           1.96 * np.std(res_acc) / np.sqrt(
                                                               args.num_test_task * args.meta_batch_size)))

    return np.mean(res_acc)

def main():
    maml = MAML(args).cuda()

    if args.resume == 1 and args.train == 1:
        model_file = '{0}/{2}/model{1}'.format(args.logdir, args.test_epoch, exp_string)
        print('model_file',model_file)
        maml.load_state_dict(torch.load(model_file))

    meta_optimiser = torch.optim.Adam(list(maml.parameters()),
                                      lr=args.meta_lr, weight_decay=args.weight_decay)

    if args.train == 1:
        train(args, maml, meta_optimiser)
    else:
        best_acc = 0
        for i in range(100):
            model_file = '{0}/{2}/model{1}'.format(args.logdir, args.test_epoch+500*i, exp_string)
            print('test_model_file',model_file)
            maml.load_state_dict(torch.load(model_file))

            acc = test(args, maml, args.test_epoch+500*i)
            if acc> best_acc:
                best_acc = acc
            print('Best_acc',best_acc)


if __name__ == '__main__':
    main()