import argparse
import random
import numpy as np
import torch
import os
from data_generator import TM
from maml import MAML
# from protonet import Protonet

parser = argparse.ArgumentParser(description='MLTI')
parser.add_argument('--datasource', default='TM', type=str,
                    help='TM')
parser.add_argument('--num_classes', default=5, type=int,
                    help='number of classes used in classification (e.g. 5-way classification).')
parser.add_argument('--test_epoch', default=-1, type=int, help='test epoch, only work when test start')

## Training options
parser.add_argument('--metatrain_iterations', default=50000, type=int,
                    help='number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
parser.add_argument('--meta_batch_size', default=4, type=int, help='number of tasks sampled per meta-update')
parser.add_argument('--update_lr', default=0.01, type=float, help='inner learning rate')
parser.add_argument('--meta_lr', default=0.001, type=float, help='the base learning rate of the generator')
parser.add_argument('--num_updates', default=5, type=int, help='num_updates in maml')
parser.add_argument('--num_updates_test', default=100, type=int, help='num_updates in maml')
parser.add_argument('--update_batch_size', default=1, type=int,
                    help='number of examples used for inner gradient update (K for K-shot learning).')
parser.add_argument('--update_batch_size_eval', default=15, type=int,
                    help='number of examples used for inner gradient test (K for K-shot learning).')
parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')

## Logging, saving, and testing options
parser.add_argument('--logdir', default='xxx', type=str,
                    help='directory for summaries and checkpoints.')
parser.add_argument('--datadir', default='../tabula_muris', type=str, help='directory for datasets.')
parser.add_argument('--resume', default=0, type=int, help='resume training if there is a model available')
parser.add_argument('--train', default=0, type=int, help='True to train, False to test.')
parser.add_argument('--mix', default=0, type=int, help='use mixup or not')
parser.add_argument('--trial', default=0, type=int, help='trail for each layer')
parser.add_argument('--ratio', default=0.2, type=float, help='the ratio of meta-training tasks')

args = parser.parse_args()
print(args)
assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True

random.seed(1)
np.random.seed(2)

exp_string = 'ATU' + '.data_' + str(args.datasource) + '.cls_' + str(args.num_classes) + '.mbs_' + str(
    args.meta_batch_size) + '.ubs_' + str(
    args.update_batch_size) + '.metalr' + str(args.meta_lr)

if args.mix:
    exp_string += '.mix'
if args.trial > 0:
    exp_string += '.trial{}'.format(args.trail)
print(exp_string)


def train(args, maml, optimiser):
    Print_Iter = 100
    Save_Iter = 100
    print_loss, print_acc = 0.0, 0.0

    dataloader = TM(args, 'train')

    if not os.path.exists(args.logdir + '/' + exp_string + '/'):
        os.makedirs(args.logdir + '/' + exp_string + '/')

    for step, (x_spt, y_spt, x_qry, y_qry, MB) in enumerate(dataloader):
        if step > args.metatrain_iterations:
            break
        # print('xpt',x_spt.shape)
        # print('MB',MB.shape)
        x_spt, y_spt, x_qry, y_qry, MB = x_spt.squeeze(0).cuda(), y_spt.squeeze(0).cuda(), \
                                     x_qry.squeeze(0).cuda(), y_qry.squeeze(0).cuda(), MB.squeeze(0).cuda()
        task_losses = []
        task_acc = []

        for meta_batch in range(args.meta_batch_size):
            if args.mix:
                loss_val, acc_val = maml.forward_ours(x_spt[meta_batch], y_spt[meta_batch],
                                                              x_qry[meta_batch],
                                                              y_qry[meta_batch],
                                                              MB[meta_batch])
            else:
                loss_val, acc_val = maml.forward(x_spt[meta_batch], y_spt[meta_batch],
                                                            x_qry[meta_batch],
                                                            y_qry[meta_batch])
            # else:
            #     loss_val, acc_val = protonet(x_spt[meta_batch], y_spt[meta_batch], x_qry[meta_batch], y_qry[meta_batch])
            task_losses.append(loss_val.squeeze())
            task_acc.append(acc_val)

        meta_batch_loss = torch.stack(task_losses).mean()
        meta_batch_acc = torch.stack(task_acc).mean()

        optimiser.zero_grad()
        meta_batch_loss.backward()
        optimiser.step()

        if step != 0 and step % Print_Iter == 0:
            print('{}, {}, {}'.format(step, print_loss, print_acc))
            print_loss, print_acc = 0.0, 0.0
        else:
            print_loss += meta_batch_loss / Print_Iter
            print_acc += meta_batch_acc / Print_Iter

        if step != 0 and step % Save_Iter == 0:
            torch.save(maml.state_dict(),
                       '{0}/{2}/model{1}'.format(args.logdir, step, exp_string))


def test(args, maml, dataloader):
    # protonet.eval()
    # maml.eval()
    res_acc = []
    args.meta_batch_size = 1
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataloader):
        # print('step',step)
        if step > 600:
            break
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to("cuda"), y_spt.squeeze(0).to("cuda"), \
                                     x_qry.squeeze(0).to("cuda"), y_qry.squeeze(0).to("cuda")
        _, acc_val = maml(x_spt, y_spt, x_qry, y_qry)
        res_acc.append(acc_val.item())

    res_acc = np.array(res_acc)

    print('acc is {}, ci95 is {}'.format(np.mean(res_acc), 1.96 * np.std(res_acc) / np.sqrt(
                                                               600 * args.meta_batch_size)))
    return np.mean(res_acc)

def main():
    maml = MAML(args).cuda()

    if args.resume == 1 and args.train == 1:
        model_file = '{0}/{2}/model{1}'.format(args.logdir, args.test_epoch, exp_string)
        print(model_file)
        maml.load_state_dict(torch.load(model_file))

    meta_optimiser = torch.optim.Adam(list(maml.parameters()),
                                      lr=args.meta_lr, weight_decay=args.weight_decay)

    if args.train == 1:
        train(args, maml, meta_optimiser)
    else:
        best_acc = 0
        print('testing!')
        random.seed(1)
        np.random.seed(2)
        dataloader = TM(args, 'test')
        for i in range(100):
            model_file = '{0}/{2}/model{1}'.format(args.logdir, args.test_epoch+100*i, exp_string)
            print('test_model_file',model_file)
            maml.load_state_dict(torch.load(model_file))
            acc = test(args, maml,dataloader)
            if acc>best_acc:
                best_acc = acc
                print('best_acc',acc)


if __name__ == '__main__':
    main()
