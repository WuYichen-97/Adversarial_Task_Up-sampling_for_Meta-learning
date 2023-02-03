import argparse
import random
import numpy as np
import torch
import os
from data_generator import miniImagenet
from maml import MAML
from TaskGenerator_BCE import TaskGenerator
#from new_structure1 import AutoEncoder
from ATU import AutoEncoder
from ATU5 import AutoEncoder as AutoEncoder5
from layers import SinkhornDistance
import random


parser = argparse.ArgumentParser(description='MetaMix')
parser.add_argument('--datasource', default='miniimagenet', type=str,
                    help='sinusoid or omniglot or miniimagenet or mixture')
parser.add_argument('--num_classes', default=5, type=int,
                    help='number of classes used in classification (e.g. 5-way classification).')
parser.add_argument('--num_test_task', default=600, type=int, help='number of test tasks.')
parser.add_argument('--test_epoch', default=17000, type=int, help='test epoch, only work when test start')

## Training options
parser.add_argument('--metatrain_iterations', default=50000, type=int,
                    help='number of metatraining iterations.')
parser.add_argument('--meta_batch_size', default=4, type=int, help='number of tasks sampled per meta-update')
parser.add_argument('--update_lr', default=0.01, type=float, help='inner learning rate')
parser.add_argument('--meta_lr', default=0.001, type=float, help='the base learning rate of the generator')
parser.add_argument('--num_updates', default=5, type=int, help='num_updates in maml')
parser.add_argument('--num_updates_test', default=5, type=int, help='num_updates in maml')
parser.add_argument('--update_batch_size', default=1, type=int,
                    help='number of examples used for inner gradient update (K for K-shot learning).')
parser.add_argument('--update_batch_size_eval', default=15, type=int,
                    help='number of examples used for inner gradient test (K for K-shot learning).')
parser.add_argument('--num_filters', default=32, type=int,
                    help='number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
## Logging, saving, and testing options
parser.add_argument('--logdir', default='xxx', type=str,
                    help='directory for summaries and checkpoints.')
parser.add_argument('--datadir', default='xxx', type=str, help='directory for datasets.')
parser.add_argument('--resume', default=0, type=int, help='resume training if there is a model available')
parser.add_argument('--train', default=1, type=int, help='True to train, False to test.')
parser.add_argument('--test_set', default=1, type=int,
                    help='Set to true to test on the the test set, False for the validation set.')
parser.add_argument('--shuffle', default=False, action='store_true', help='use channel shuffle or not')
parser.add_argument('--mix', default=True, action='store_true', help='use mixup or not')
parser.add_argument('--trial', default=0, type=int, help='trial')
parser.add_argument('--method', default='ours', type=str, help='choose the method [yao,ours]')
args = parser.parse_args()
print(args)

assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True

random.seed(1)
np.random.seed(2)

exp_string = f'ATU.data_{args.datasource}.cls_{args.num_classes}.mbs_{args.meta_batch_size}.ubs_{args.update_batch_size}.metalr_{args.meta_lr}.innerlr_{args.update_lr}'
if args.num_filters != 64:
    exp_string += '.hidden' + str(args.num_filters)
if args.mix:
    exp_string += '.mix'
if args.shuffle:
    exp_string += '.shuffle'
if args.trial > 0:
    exp_string += '.trial_{}'.format(args.trial)
exp_string += 'maml3'#str(args.method)+'xs'#+'True'#str(args.method)+'64'#+str(False)
print(exp_string)

def construct_task_pool(task_pool,MB_feats,N):
    MB_feats = MB_feats.reshape(5,3,84,84)
    index = torch.randperm(5)[:3]
    MB_feats = MB_feats[index] 
#    N = 16
    temp0 = np.random.beta(3,5,size=2)
    temp0.sort()
    temp = torch.tensor(temp0[0]).cuda()
    temp1 = torch.tensor(temp0[1]).cuda()
    coarse0 = torch.zeros(2*N,5,3,84,84)
    for i in range(N):
        lamda = temp
        coarse0[i] = lamda*task_pool[i] + (1-lamda)*MB_feats[0].unsqueeze(0)     
    for i in range(N,2*N):
        lamda = temp1
        coarse0[i] = lamda*task_pool[i-N] + (1-lamda)*MB_feats[0].unsqueeze(0)  

    coarse1 = torch.zeros(2*N,5,3,84,84)
    for i in range(N):
        lamda = temp
        coarse1[i] = lamda*task_pool[i] + (1-lamda)*MB_feats[1].unsqueeze(0)       
    for i in range(N,2*N):
        lamda = temp1
        coarse1[i] = lamda*task_pool[i-N] + (1-lamda)*MB_feats[1].unsqueeze(0)

    coarse2 = torch.zeros(2*N,5,3,84,84)
    for i in range(N):
        lamda = temp
        coarse2[i] = lamda*task_pool[i] + (1-lamda)*MB_feats[2].unsqueeze(0)    
    for i in range(N,2*N):
        lamda = temp1
        coarse2[i] = lamda*task_pool[i-N] + (1-lamda)*MB_feats[2].unsqueeze(0)
        
    return coarse0,coarse1,coarse2,MB_feats

def COindex(T1,T2):
    T1 = T1.reshape(len(T1),-1)
    T2 = T2.reshape(len(T2),-1)
    T2_copy = torch.sum(T2, dim=1, dtype=torch.float64)
    for i in range(len(T1)):
        d = (T1[i]-T2)**2 
        _, index = torch.min(torch.sum(d,1),0)
        if i == 0:
            out_index = index.unsqueeze(0) #.unsqueeze(0)
        else:
            index1 = (T2_copy == torch.sum(T2[index],0,dtype=torch.float64)).nonzero(as_tuple=True)[0]
            out_index = torch.cat((out_index,index1),0)
    return out_index.cpu().numpy().tolist()

def select_task_pool(sinkhorn,task_pool,coarse0,coarse1,coarse2,MB,N):
    # task_pool #(16,5,3,84,84)  
    # coarse #(32,3,84,84)
    class_ori = torch.mean(task_pool.reshape(N,5,3,84,84),dim=0)   # (5,3,84,84)
    index0 = COindex(class_ori,MB)
    gt0 = coarse0.permute(1,0,2,3,4).unsqueeze(0) #(1,5,32,3,84,84)
    gt1 = coarse1.permute(1,0,2,3,4).unsqueeze(0)
    gt2 = coarse2.permute(1,0,2,3,4).unsqueeze(0)
    gt = torch.cat((gt0,gt1,gt2),dim=0) #(3,5,48,3,84,84)
    sel_gt = torch.zeros(5,2*N,3,84,84) #(5,48,3,84,84)
    for i in range(5):
        sel_gt[i] = gt[index0[i],i]
    return sel_gt.permute(1,0,2,3,4).cuda()

    
def train(args, maml, optimiser):
    Print_Iter = 100
    Save_Iter = 500
    print_loss, print_acc, print_loss_support = 0.0, 0.0, 0.0
    dataloader = miniImagenet(args, 'train')
    if not os.path.exists(args.logdir + '/' + exp_string + '/'):
        os.makedirs(args.logdir + '/' + exp_string + '/')
    save_path = 'xxx' + '/' + 'PCN-3' + '/'
    if not os.path.exists(save_path):
            os.makedirs(save_path)  
    
    if args.update_batch_size ==1:     
       TaskGen = TaskGenerator(AutoEncoder,1)
    else:
       TaskGen = TaskGenerator(AutoEncoder5,5) 
    sinkhorn = SinkhornDistance(eps=0.01, max_iter=100, reduction='mean')
    if args.method == 'ours':
        count_step = 0
        for step, (x_spt, y_spt, x_qry, y_qry, MB_x) in enumerate(dataloader):
            if step > args.metatrain_iterations:
                break
            x_spt, y_spt, x_qry, y_qry, MB_x = x_spt.squeeze(0).cuda(), y_spt.squeeze(0).cuda(), \
                                         x_qry.squeeze(0).cuda(), y_qry.squeeze(0).cuda(), MB_x.squeeze(0).cuda()
            task_losses = []
            task_acc = []                
            for i in range(args.meta_batch_size):
                
                if args.update_batch_size ==1:     
                    supp, ys, query, yq, MB = x_spt[i], y_spt[i], x_qry[i], y_qry[i], MB_x[i]
                    # supp_temp = supp.reshape(5,5,3,84,84).permute(1,0,2,3,4)
                    query_temp = query.reshape(5,15,3,84,84).permute(1,0,2,3,4) #(15,5,3,84,84)
                    task_ori_pool = torch.cat((supp.unsqueeze(0),query_temp),dim=0) #(16,5,3,84,84) 
                    coarse0, coarse1, coarse2, MB_feats = construct_task_pool(task_ori_pool,MB,16)   
                    
                    # mix_c = random.randint(0,1)
                    if random.random() <= 0.8:  # Augmentation Ratio 
                    # if mix_c == 1:  
                        partial_input = task_ori_pool.reshape(80,3,84,84).cuda()
                        gt = select_task_pool(sinkhorn,task_ori_pool,coarse0,coarse1,coarse2,MB_feats,16)
                        supp_aug, query_aug = TaskGen.Upsampling(maml,ys,yq,partial_input,gt,MB_feats,step)
                        loss_val, acc_val = maml.forward_ours(supp,ys,supp_aug,query,query_aug,yq)
                    else:
                        loss_val, acc_val = maml.forward_metamix(x_spt[i], y_spt[i],
                                                                    x_qry[i],
                                                                    y_qry[i])    
                else:
                    supp, ys, query, yq, MB = x_spt[i], y_spt[i], x_qry[i], y_qry[i], MB_x[i]
                    supp_temp = supp.reshape(5,5,3,84,84).permute(1,0,2,3,4)
                    query_temp = query.reshape(5,15,3,84,84).permute(1,0,2,3,4) #(15,5,3,84,84)
                    task_ori_pool = torch.cat((supp_temp,query_temp),dim=0) #(16,5,3,84,84) 
                    coarse0, coarse1, coarse2, MB_feats = construct_task_pool(task_ori_pool,MB,20)   
                    
                    # mix_c = random.randint(0,1)
                    if random.random() <= 0.5:  # Augmentation Ratio 
                    # if mix_c == 1:  
                        partial_input = task_ori_pool.reshape(100,3,84,84).cuda()
                        gt = select_task_pool(sinkhorn,task_ori_pool,coarse0,coarse1,coarse2,MB_feats,20)
                        supp_aug, query_aug = TaskGen.Upsampling(maml,ys,yq,partial_input,gt,MB_feats,step)
                        loss_val, acc_val = maml.forward_ours(supp,ys,supp_aug,query,query_aug,yq)
                    else:
                        loss_val, acc_val = maml.forward_metamix(x_spt[i], y_spt[i],
                                                                    x_qry[i],
                                                                    y_qry[i])  
                    
                # task_losses.append(loss_val)
                task_losses.append(loss_val.squeeze())
                task_acc.append(acc_val)  
            meta_batch_loss = torch.stack(task_losses).mean()
            meta_batch_acc = torch.stack(task_acc).mean()
    
            optimiser.zero_grad()
            meta_batch_loss.backward()
            optimiser.step()               
                
            count_step += 1
            if count_step != 0 and count_step % (Print_Iter) == 0:
                print('iter: {}, loss_all: {}, acc: {}'.format(count_step, print_loss, print_acc))
                print_loss, print_acc = 0.0, 0.0
            else:
                print_loss += meta_batch_loss / Print_Iter
                print_acc += meta_batch_acc / Print_Iter
            if count_step != 0 and count_step % Save_Iter == 0:
                torch.save(maml.state_dict(),
                           '{0}/{2}/model{1}'.format(args.logdir, count_step, exp_string))



def val(args, maml, test_epoch):
    res_acc = []
    args.meta_batch_size = 1
    dataloader = miniImagenet(args, 'val')
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataloader):
        if step > args.num_test_task:
            break
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).cuda(), y_spt.squeeze(0).cuda(), \
                                     x_qry.squeeze(0).cuda(), y_qry.squeeze(0).cuda()
                                     
        _, acc_val = maml(x_spt, y_spt, x_qry, y_qry)
        res_acc.append(acc_val.item())
    res_acc = np.array(res_acc)
    return np.mean(res_acc)

def test(args, maml, test_epoch):
    res_acc = []
    args.meta_batch_size = 1
    dataloader = miniImagenet(args, 'test')
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
        model_file = '{0}/{2}/model{1}'.format(args.logdir, args.test_epoch, exp_string)
        print('test_model_file',model_file)
        maml.load_state_dict(torch.load(model_file))
        test(args, maml, args.test_epoch)

if __name__ == '__main__':
    main()
