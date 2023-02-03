import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import argparse
class miniImagenet(Dataset):

    def __init__(self, args, mode):
        super(miniImagenet, self).__init__()
        self.args = args
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes  # n-way
        self.k_shot = args.update_batch_size  # k-shot
        self.k_query = args.update_batch_size_eval  # for evaluation
        self.set_size = self.n_way * self.k_shot  # num of samples per set
        self.query_size = self.n_way * self.k_query  # number of samples per set for evaluation
        self.mode = mode
        if mode == 'train':
            self.data_file = '../miniImagenet/mini_imagenet_train.pkl'.format(args.datadir)
        elif mode == 'test':
            self.data_file = '../miniImagenet/mini_imagenet_test.pkl'.format(args.datadir)
        elif mode == 'val':
            self.data_file = '../miniImagenet/mini_imagenet_val.pkl'.format(args.datadir)

        self.data = pickle.load(open(self.data_file, 'rb'))
        
        self.method = args.method
        self.MB_samples_per_class = 1
        self.MB_set_size = 5 * self.MB_samples_per_class

        self.data = torch.tensor(np.transpose(self.data / np.float32(255), (0, 1, 4, 2, 3)))
        self.fixed_class_id = np.array([26, 59, 14, 16,  17,  52, 8, 39,  46, 32, 20, 57])


    def __len__(self):
        return self.args.metatrain_iterations*self.args.meta_batch_size

    def __getitem__(self, index):
        self.classes_idx = np.arange(self.data.shape[0])
        self.samples_idx = np.arange(self.data.shape[1])
        
        support_x = torch.FloatTensor(torch.zeros((self.args.meta_batch_size, self.set_size, 3, 84, 84)))
        query_x = torch.FloatTensor(torch.zeros((self.args.meta_batch_size, self.query_size, 3, 84, 84)))
        MB_x = torch.FloatTensor(torch.zeros((self.args.meta_batch_size, self.MB_set_size, 3, 84, 84)))

        support_y = np.zeros([self.args.meta_batch_size, self.set_size])
        query_y = np.zeros([self.args.meta_batch_size, self.query_size])

        for meta_batch_id in range(self.args.meta_batch_size):
            if self.mode =='train':
                self.choose_classes = np.random.choice(self.fixed_class_id, size=self.nb_classes, replace=False)
                self.Memory_Bank_classes = (list(set(self.fixed_class_id).difference(self.choose_classes))) # MB different class with meta-train
                
                for j in range(self.nb_classes):
                    np.random.shuffle(self.samples_idx)
                    choose_samples = self.samples_idx[:self.nb_samples_per_class]
                    #chosse_samples_mb = self.samples_idx[:self.MB_samples_per_class]
                    
                    support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = self.data[
                        self.choose_classes[
                            j], choose_samples[
                                :self.k_shot], ...]
                    query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = self.data[
                        self.choose_classes[
                            j], choose_samples[
                                self.k_shot:], ...]
                    
                    MB_x[meta_batch_id][j * self.MB_samples_per_class:(j + 1) * self.MB_samples_per_class] = self.data[
                        self.Memory_Bank_classes[
                            j], choose_samples[
                                :self.MB_samples_per_class], ...] 
                   
                    support_y[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = j
                    query_y[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = j            
        
            else:
                self.choose_classes = np.random.choice(self.classes_idx, size=self.nb_classes, replace=False)
                            
                for j in range(self.nb_classes):
                    np.random.shuffle(self.samples_idx)
                    choose_samples = self.samples_idx[:self.nb_samples_per_class]
                    support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = self.data[
                        self.choose_classes[
                            j], choose_samples[
                                :self.k_shot], ...]
                    query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = self.data[
                        self.choose_classes[
                            j], choose_samples[
                                self.k_shot:], ...]
                    support_y[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = j
                    query_y[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = j
        if self.mode =='train':
            return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y), MB_x        
        else:
            return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='MetaMix')
    parser.add_argument('--datasource', default='miniimagenet', type=str,
                        help='sinusoid or omniglot or miniimagenet or mixture')
    parser.add_argument('--num_classes', default=5, type=int,
                        help='number of classes used in classification (e.g. 5-way classification).')
    parser.add_argument('--num_test_task', default=600, type=int, help='number of test tasks.')
    parser.add_argument('--test_epoch', default=-1, type=int, help='test epoch, only work when test start')
    
    ## Training options
    parser.add_argument('--metatrain_iterations', default=15000, type=int,
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
    parser.add_argument('--train', default=1, type=int, help='True to train, False to test.')
    parser.add_argument('--test_set', default=1, type=int,
                        help='Set to true to test on the the test set, False for the validation set.')
    parser.add_argument('--shuffle', default=False, action='store_true', help='use channel shuffle or not')
    parser.add_argument('--mix', default=False, action='store_true', help='use mixup or not')
    parser.add_argument('--trial', default=0, type=int, help='trial')
    parser.add_argument('--method', default='ours', type=str, help='choose the method [yao,ours]')
    
    args = parser.parse_args()

    # dataloader = miniImagenet(args, 'train')
    # for step, (x_spt, y_spt, x_qry, y_qry, MB_x)  in enumerate(dataloader):
    #     for i in range(4):
    #         print('x_spt',x_spt[i])
    #         print('x_spt',x_spt[i].shape)

    dataloader = miniImagenet(args, 'test')
    for step, (x_spt, y_spt, x_qry, y_qry)  in enumerate(dataloader):
        for i in range(4):
            print('x_spt',x_spt[i])
            print('x_spt',x_spt[i].shape)
        # print('step',step)
        












