import argparse
import random
import numpy as np
import torch
import os
# from data_generator1 import miniImagenet, miniImagenet1
#from data_generator import miniImagenet#, miniImagenet1
from maml import MAML
# from TaskGenerator_BCE3 import TaskGenerator
from new_structure import AutoEncoder
from Dataloader_MB import ISIC 
import torch.utils.data as data

import random
parser = argparse.ArgumentParser(description='MetaMix')
parser.add_argument('--datasource', default='miniimagenet', type=str,
                    help='sinusoid or omniglot or miniimagenet or mixture')
parser.add_argument('--num_classes', default=5, type=int,
                    help='number of classes used in classification (e.g. 5-way classification).')
parser.add_argument('--num_test_task', default=600, type=int, help='number of test tasks.')
parser.add_argument('--test_epoch', default=1000, type=int, help='test epoch, only work when test start')

## Training options
parser.add_argument('--metatrain_iterations', default=50000, type=int,
                    help='number of metatraining iterations.')
parser.add_argument('--meta_batch_size', default=4, type=int, help='number of tasks sampled per meta-update')
parser.add_argument('--update_lr', default=0.01, type=float, help='inner learning rate')
parser.add_argument('--meta_lr', default=0.001, type=float, help='the base learning rate of the generator')
parser.add_argument('--num_updates', default=5, type=int, help='num_updates in maml')
parser.add_argument('--num_updates_test', default=10, type=int, help='num_updates in maml')
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
parser.add_argument('--train', default=0, type=int, help='True to train, False to test.')
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

exp_string = f'MetaMix.data_{args.datasource}.cls_{args.num_classes}.mbs_{args.meta_batch_size}.ubs_{args.update_batch_size}.metalr_{args.meta_lr}.innerlr_{args.update_lr}'
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


def COindex(T1,T2):
    T1 = T1.reshape(len(T1),-1)
    T2 = T2.reshape(len(T2),-1)
    T2_copy = torch.sum(T2, dim=1, dtype=torch.float64)
    for i in range(len(T1)):
        d = (T1[i]-T2)**2 
        _, index = torch.min(torch.sum(d,1),0)
        if i == 0:
            out_index = index.unsqueeze(0) #.unsqueeze(0)
            T2 = del_tensor_ele(T2, index)
        else:
            index1 = (T2_copy == torch.sum(T2[index],0,dtype=torch.float64)).nonzero(as_tuple=True)[0]
            out_index = torch.cat((out_index,index1),0)
            T2 = del_tensor_ele(T2, index)
    return out_index.cpu().numpy().tolist()


def task_Aug(task_pool,MB_feats):
    # task_pool #(16,5,32,42,42)  
    # MB_feats #(21,32,42,42)
    MB_feats = MB_feats.reshape(7,3,3,84,84)
    task_mean = torch.mean(task_pool,dim=0) #(5,3,84,84)
    MB_feats_mean =  torch.mean(MB_feats, dim=1)   #(7,3,84,84)
    index = COindex(task_mean, MB_feats_mean)  #[5]   
    MB_feats = MB_feats[index] #(5,3,3,84,84)
    
    # N = task_pool.shape[0]
    N = 8 
    coarse = torch.zeros(N,5,3,84,84)
    idx1 = int(np.random.choice(a=3,size=1))
    for i in range(N):
        temp = np.random.beta(2,2)
        # temp = np.random.beta(0.8,0.8)
        lamda = max(temp,1-temp)
        coarse[i] = lamda*task_pool[i] + (1-lamda)*MB_feats[:,idx1]      
    
    
    detail = torch.zeros(N,5,3,84,84)
    # idx1 = int(np.random.choice(a=3,size=1))
    for i in range(N):
        temp = np.random.beta(0.5,0.5)
        # temp = np.random.beta(0.8,0.8)
        lamda = max(temp,1-temp)
        detail[i] = lamda*task_pool[i] + (1-lamda)*MB_feats[:,idx1]   
    return coarse.reshape(8,5,3,84,84), detail.reshape(8,5,3,84,84)
    # return  out.reshape(16,5,3,84,84)




# task Augmentation Part
def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)
    


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
            
    TaskGen = TaskGenerator(4000, AutoEncoder)
    if args.method == 'ours':
        count_step = 0
        # best_val = 0
        for step, (x_spt, y_spt, x_qry, y_qry, MB_x) in enumerate(dataloader):
            if step > args.metatrain_iterations:
                break
            x_spt, y_spt, x_qry, y_qry, MB_x = x_spt.squeeze(0).cuda(), y_spt.squeeze(0).cuda(), \
                                         x_qry.squeeze(0).cuda(), y_qry.squeeze(0).cuda(), MB_x.squeeze(0).cuda() 
            i = 0
            supp, ys, query, yq , MB = x_spt[i], y_spt[i], x_qry[i], y_qry[i],  MB_x[i]
            query = query.reshape(5,15,3,84,84).permute(1,0,2,3,4) #(15,5,3,84,84)
            task_pool = torch.cat((supp.unsqueeze(0),query),dim=0) #(16,5,3,84,84) 
            # task_coarse_pool, task_detail_pool = task_Aug(task_pool, MB) #(32,5,3,84,84) 
            partial_input, gt, pred_coarse = TaskGen.Upsampling(maml, ys, yq, task_pool, step, Aug=False)            
            if step % 1000 == 0:
               torch.save(partial_input, save_path+'partial_input'+str(step)+'.pt')
               torch.save(gt, save_path +'gt'+str(step)+'.pt')
               torch.save(pred_coarse, save_path+'pred_coarse'+str(step)+'.pt')
               
            for j in range(int(args.meta_batch_size/4)):
                task_losses = []
                task_acc = []
                if step <= 3000:  # pre-trained PCN
                    for meta_batch in range(4):
                        loss_val, acc_val = maml.forward(x_spt[j*4+meta_batch], y_spt[j*4+meta_batch], x_qry[j*4+meta_batch], y_qry[j*4+meta_batch])
                        task_losses.append(loss_val)
                        task_acc.append(acc_val)

                    meta_batch_loss = torch.stack(task_losses).mean()
                    meta_batch_acc = torch.stack(task_acc).mean()

                    optimiser.zero_grad()
                    meta_batch_loss.backward()
                    optimiser.step()

                else:
                    for meta_batch in range(4):
                        if random.random() <= 0.6:
                            supp, query, MB = x_spt[j*4+meta_batch], x_qry[j*4+meta_batch], MB_x[j*4+meta_batch]
                            query = query.reshape(5,15,3,84,84).permute(1,0,2,3,4) #(15,5,3,84,84)
                            task_pool = torch.cat((supp.unsqueeze(0),query),dim=0) #(16,5,3,84,84) 
                            task_coarse_pool,task_detail_pool = task_Aug(task_pool, MB) #(32,5,3,84,84)            
                            supp, query = TaskGen.Test_Upsampling(task_pool, task_coarse_pool,task_detail_pool)
                            
                            if step % 3000==0:
                                torch.save(x_spt[j*4+meta_batch], save_path+'supp_gt'+str(step)+'.pt')
                                torch.save(x_qry[j*4+meta_batch], save_path+'qry_gt'+str(step)+'.pt')
                                torch.save(supp, save_path+'supp'+str(step)+'.pt')
                                torch.save(query, save_path+'query'+str(step)+'.pt')  
                            loss_val, acc_val = maml.forward(supp,ys,query,yq)

                        else:
                            loss_val, acc_val = maml.forward(x_spt[j*4+meta_batch], y_spt[j*4+meta_batch], x_qry[j*4+meta_batch], y_qry[j*4+meta_batch])
                        
                        task_losses.append(loss_val)
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
    maml.eval()
    dataloader = miniImagenet1(args, 'val')
   
    # for step, (x_spt, y_spt, x_qry, y_qry, x_spt2, x_qry2,_) in enumerate(dataloader):
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataloader):
    # for step, (x_spt, y_spt, x_qry, y_qry, MB_x) in enumerate(dataloader):
        if step > args.num_test_task:
            break
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).cuda(), y_spt.squeeze(0).cuda(), \
                                     x_qry.squeeze(0).cuda(), y_qry.squeeze(0).cuda()
                                     
        _, acc_val = maml(x_spt, y_spt, x_qry, y_qry)
        # print('acc_val', acc_val)
        res_acc.append(acc_val.item())

    res_acc = np.array(res_acc)
    return np.mean(res_acc)

def test(args, maml, test_epoch,dataloader):
    res_acc = []
    args.meta_batch_size = 1
    # maml.eval()
#    random.seed(1)
#    np.random.seed(2)  
#    dataloader = miniImagenet(args, 'test')

    for step, (x_spt, y_spt, x_qry, y_qry,_) in enumerate(dataloader):
    # for step, (x_spt, y_spt, x_qry, y_qry, MB_x) in enumerate(dataloader):
        if step > args.num_test_task:
            break
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).cuda(), y_spt.squeeze(0).cuda(), \
                                     x_qry.squeeze(0).cuda(), y_qry.squeeze(0).cuda()
                                     
        _, acc_val = maml(x_spt, y_spt, x_qry, y_qry)
        # print('acc_val', acc_val)
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
    
    meta_optimiser = torch.optim.Adam(list(filter(lambda p: p.requires_grad, maml.parameters())),
                                                      lr=args.meta_lr, weight_decay=args.weight_decay)      

    if args.train == 1:
        train(args, maml, meta_optimiser)
    else:
        random.seed(1)
        np.random.seed(2)
        isic_test = ISIC('../', mode='test', n_way=args.num_classes, k_shot=args.update_batch_size, k_query=args.update_batch_size_eval,
                batchsz=600, resize=84)
        dataloader = data.DataLoader(isic_test, 1, shuffle=False)    
                
        
        best_acc = 0
        for i in range(100):
            model_file = '{0}/{2}/model{1}'.format(args.logdir, args.test_epoch+500*i, exp_string)
            maml.load_state_dict(torch.load(model_file))
            acc = test(args, maml, args.test_epoch+500*i,dataloader)
            if acc>best_acc:
                best_acc = acc
                print('best_acc',acc)

if __name__ == '__main__':
    main()
