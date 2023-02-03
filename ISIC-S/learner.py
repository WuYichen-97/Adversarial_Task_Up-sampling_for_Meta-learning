import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.distributions import Beta
from collections import OrderedDict
import numpy as np

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv_Standard(nn.Module):
    def __init__(self, args, x_dim, hid_dim, z_dim, final_layer_size):
        super(Conv_Standard, self).__init__()
        self.args = args
        self.net = nn.Sequential(self.conv_block(x_dim, hid_dim), self.conv_block(hid_dim, hid_dim),
                                 self.conv_block(hid_dim, hid_dim), self.conv_block(hid_dim, z_dim), Flatten())
        self.dist = Beta(torch.FloatTensor([2]), torch.FloatTensor([2]))
        self.dist_ours = Beta(torch.FloatTensor([0.5]), torch.FloatTensor([0.5]))
        self.hid_dim = hid_dim

        self.logits = nn.Linear(final_layer_size, self.args.num_classes)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def functional_conv_block(self, x, weights, biases,
                              bn_weights, bn_biases, is_training):

        x = F.conv2d(x, weights, biases, padding=1)
        x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases,
                         training=is_training)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x


    def mixup_data(self, xs, xs2, lam=None):
        if not lam:
            lam = self.dist.sample().cuda()
        mixed_xs = lam * xs + (1 - lam) * xs2
        return mixed_xs, lam
    
    def mixup_data1(self, xs, xs2, lam=None):
        if not lam:
            lam = 0
        mixed_xs = lam * xs + (1 - lam) * xs2
        return mixed_xs, lam    



    def forward(self, x):
        x = self.net(x)

        x = x.view(x.size(0), -1)

        return self.logits(x)

    def functional_forward(self, x, weights, is_training=True):
        for block in range(4):
            x = self.functional_conv_block(x, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'],
                                           weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'),
                                           is_training)

        x = x.view(x.size(0), -1)

        x = F.linear(x, weights['logits.weight'], weights['logits.bias'])

        return x

    def functional_forward_fixed(self, x, fixed_weights, weights, is_training=True):
        x = self.functional_conv_block(x, fixed_weights[f'net.{0}.0.weight'], fixed_weights[f'net.{0}.0.bias'],
                               fixed_weights.get(f'net.{0}.1.weight'), fixed_weights.get(f'net.{0}.1.bias'),
                               is_training)
        for block in range(1,4):
            x = self.functional_conv_block(x, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'],
                                           weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'),
                                           is_training)

        x = x.view(x.size(0), -1)

        x = F.linear(x, weights['logits.weight'], weights['logits.bias'])

        return x


    
    def functional_forward_classifier(self, x, weights, sel_layer, is_training=True):
        for block in range(sel_layer,4):
            x = self.functional_conv_block(x, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'],
                                           weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'),
                                           is_training)
        x = x.view(x.size(0), -1)
        x = F.linear(x, weights['logits.weight'], weights['logits.bias'])  
        return x    
    
    def functional_forward_features(self, x, sel_layer, weights, is_training=True):
        #for block in range(4):
        for block in range(sel_layer):
            x = self.functional_conv_block(x, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'],
                                           weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'),
                                           is_training)
        return x

    def channel_shuffle(self, hidden, label, shuffle_dict, shuffle_channel_id):
        concept_idx = [0, 6, 11, 16, 22, 27, 32]

        new_data = []

        start = concept_idx[shuffle_channel_id]
        end = concept_idx[shuffle_channel_id + 1]

        for i in range(self.args.num_classes):
            cur_class_1 = hidden[label == i]
            cur_class_2 = hidden[label == shuffle_dict[i]]

            new_data.append(
                torch.cat((cur_class_1[:, :start], cur_class_2[:, start:end], cur_class_1[:, end:]), dim=1))

        new_data = torch.cat(new_data, dim=0)

        indexes = torch.randperm(new_data.shape[0])

        new_data = new_data[indexes]
        new_label = label[indexes]

        return new_data, new_label

    def forward_metamix_supp1(self, hidden_support, hidden_support2, weights, is_training=True):
        sel_layer = random.randint(0, 3)
        flag = 0
        for layer in range(4):
            if layer==sel_layer:
                hidden_support, lam = self.mixup_data1(hidden_support, hidden_support2)
                flag=1
            if not flag:
                hidden_support2 = self.functional_conv_block(hidden_support2, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                is_training)     
            hidden_support = self.functional_conv_block(hidden_support, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                            weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                            is_training)     
                
        mix_supp = hidden_support.view(hidden_support.size(0), -1)
        logits  = F.linear(mix_supp , weights['logits.weight'], weights['logits.bias'])
        return logits, sel_layer, lam

    def forward_metamix_query1(self, hidden_support, hidden_support2, weights,  lamda, sel_layer, is_training=True):
        sel_layer = sel_layer
        flag = 0
        for layer in range(4):
            if layer==sel_layer:
                hidden_support,_ = self.mixup_data1(hidden_support, hidden_support2, lamda)
                flag=1
            if not flag:
                hidden_support2 = self.functional_conv_block(hidden_support2, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                is_training)     
            hidden_support = self.functional_conv_block(hidden_support, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                            weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                            is_training)                     
        hidden_support = hidden_support.view(hidden_support.size(0), -1)
        logits  = F.linear(hidden_support , weights['logits.weight'], weights['logits.bias'])
        
        return logits






    def forward_metamix_supp(self, hidden_support, hidden_support2, weights, is_training=True):
        sel_layer = random.randint(0, 3)
        flag = 0
        for layer in range(4):
            if layer==sel_layer:
                hidden_support, lam = self.mixup_data(hidden_support, hidden_support2)
                flag=1
            if not flag:
                hidden_support2 = self.functional_conv_block(hidden_support2, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                is_training)     
            hidden_support = self.functional_conv_block(hidden_support, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                            weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                            is_training)     
                
        mix_supp = hidden_support.view(hidden_support.size(0), -1)
        logits  = F.linear(mix_supp , weights['logits.weight'], weights['logits.bias'])
        return logits, sel_layer, lam

    def forward_metamix_query(self, hidden_support, hidden_support2, weights,  lamda, sel_layer, is_training=True):
        sel_layer = sel_layer
        flag = 0
        for layer in range(4):
            if layer==sel_layer:
                hidden_support,_ = self.mixup_data(hidden_support, hidden_support2, lamda)
                flag=1
            if not flag:
                hidden_support2 = self.functional_conv_block(hidden_support2, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                is_training)     
            hidden_support = self.functional_conv_block(hidden_support, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                            weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                            is_training)                     
        hidden_support = hidden_support.view(hidden_support.size(0), -1)
        logits  = F.linear(hidden_support , weights['logits.weight'], weights['logits.bias'])
        
        return logits


    def COindex_ours(self, T1, T2):
        """
        T1: query
        T2: databse
        """
        T1 = T1.reshape(len(T1),-1).detach()
        T2 = T2.reshape(len(T2),-1).detach()
        dist = torch.cdist(T1.unsqueeze(0), T2.unsqueeze(0)).squeeze(0)
        d_max = torch.max(dist)
        for i in range(len(T1)):
            d = dist[i]
            _, index = torch.min(d, 0)
            if i == 0:
                out_index = index.unsqueeze(0)#.unsqueeze(0)
            else:
                out_index = torch.cat((out_index, index.unsqueeze(0)), 0)
            dist[:, index] = d_max + 1

        return out_index


    def mixup_data_ours(self, xs, xs2, index = None, lam=None):
        set_size = xs.shape[0]
        xs_mean = xs.clone()
        xs_mean = xs_mean.reshape(5,int(set_size/5),xs_mean.shape[1],xs_mean.shape[2],xs_mean.shape[3])
        xs_mean = torch.mean(xs_mean, dim=1)
        
        xs2 = xs2.reshape(7,3,xs2.shape[1],xs2.shape[2],xs2.shape[3])
        if index == None:
            xs2_mean = torch.mean(xs2,dim=1)
            index = self.COindex_ours(xs_mean, xs2_mean)
            
            
        xs = xs.reshape(5,int(set_size/5),xs.shape[1],xs.shape[2],xs.shape[3])    
        xs2 = xs2[index.long()]
        
        
        'Each class have the same lam for both supp & query'
        if not lam:
            # lam = []
            # for _ in range(5):
            #     lam.append(self.dist_ours.sample().cuda())        
            lam = self.dist_ours.sample().cuda()
        
        mix_feature = torch.zeros_like(xs)
        for i in range(5):
            for j in range(int(set_size/5)):
                lam_i = lam
                idx = int(np.random.choice(a=3,size=1))
                mix_feature[i,j] = lam_i*xs[i,j]+ (1-lam_i)*xs2[i,idx]
        mix_feature = mix_feature.reshape(set_size,xs.shape[2],xs.shape[3],xs.shape[4])    
        return mix_feature, lam, index



    def mixup_data_metamix(self, xs, ys, xq, yq):
        # print('xs',xs.shape)
        # print('xq',xq.shape)
        # print('ys',ys.shape)
        # print('yq',yq.shape)
        
        query_size = xq.shape[0]

        shuffled_index = torch.randperm(query_size)

        xs = xs[shuffled_index]
        ys = ys[shuffled_index]
        lam = self.dist.sample().cuda()
        mixed_x = lam * xq + (1 - lam) * xs

        return mixed_x, yq, ys, lam
    
    def forward_metamix(self, hidden_support, label_support, hidden_query, label_query, weights, is_training=True):
    
            sel_layer = random.randint(0, 3)
            flag = 0
    
            for layer in range(4):
                if layer==sel_layer:
                    hidden_query, reweighted_query, reweighted_support, lam = self.mixup_data_metamix(hidden_support, label_support,
                                                                                           hidden_query, label_query)
                    flag=1
    
                if not flag:
                    hidden_support = self.functional_conv_block(hidden_support, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                    weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                    is_training)
                hidden_query = self.functional_conv_block(hidden_query, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                is_training)
    
            hidden4_query = hidden_query.view(hidden_query.size(0), -1)
    
            x = F.linear(hidden4_query, weights['logits.weight'], weights['logits.bias'])
    
            return x, reweighted_query, reweighted_support, lam



    def forward_metamix_supp_ours(self, hidden_support, hidden_support2, weights, is_training=True):
        sel_layer = random.randint(0, 3)
        flag = 0
        
        for layer in range(4):
            if layer==sel_layer:
                hidden_support, lam, index = self.mixup_data_ours(hidden_support, hidden_support2)
                flag=1
            if not flag:
                hidden_support2 = self.functional_conv_block(hidden_support2, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                is_training)     
            hidden_support = self.functional_conv_block(hidden_support, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                            weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                            is_training)     
                
        mix_supp = hidden_support.view(hidden_support.size(0), -1)
        logits  = F.linear(mix_supp , weights['logits.weight'], weights['logits.bias'])
  
        return logits, sel_layer, lam, index

    def forward_metamix_query_ours(self, hidden_support, hidden_support2, weights,  lamda, sel_layer, index, is_training=True):
        sel_layer = sel_layer
        flag = 0
        for layer in range(4):
            if layer==sel_layer:
                hidden_support,lam,_ = self.mixup_data_ours(hidden_support, hidden_support2, index, lamda)
                flag=1
            if not flag:
                hidden_support2 = self.functional_conv_block(hidden_support2, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                is_training)     
            hidden_support = self.functional_conv_block(hidden_support, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                            weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                            is_training)                     
        hidden_support = hidden_support.view(hidden_support.size(0), -1)
        logits  = F.linear(hidden_support , weights['logits.weight'], weights['logits.bias'])
        
        return logits
    
    def forward_gen_ours(self, hidden_support, hidden_support2, weights, sel_layer, is_training=True):
        
        flag = 0
        for layer in range(4):
            if layer==sel_layer:
                hidden_support = hidden_support2
                # hidden_support,lam,_ = self.mixup_data_ours(hidden_support, hidden_support2, index, lamda)
                flag=1
            if not flag:
                hidden_support2 = self.functional_conv_block(hidden_support2, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                is_training)     
            hidden_support = self.functional_conv_block(hidden_support, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                            weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                            is_training)                     
        hidden_support = hidden_support.view(hidden_support.size(0), -1)
        logits  = F.linear(hidden_support , weights['logits.weight'], weights['logits.bias'])
        
        return logits




    def functional_forward_cf(self, hidden, label, sel_layer, shuffle_list, shuffle_channel_id, weights,
                                           is_training=True):

        label_new = label

        for layer in range(4):
            if layer == sel_layer:
                hidden, label_new = self.channel_shuffle(hidden, label, shuffle_list, shuffle_channel_id)

            hidden = self.functional_conv_block(hidden, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                is_training)

        hidden = hidden.view(hidden.size(0), -1)

        x = F.linear(hidden, weights['logits.weight'], weights['logits.bias'])

        return x, label_new

    def mix_cf(self, hidden_support, label_new_support, hidden_query, label_new_query, shuffle_list,
               shuffle_channel_id):

        hidden_support, label_new_support = self.channel_shuffle(hidden_support, label_new_support, shuffle_list,
                                                                 shuffle_channel_id)
        hidden_query, label_new_query = self.channel_shuffle(hidden_query, label_new_query, shuffle_list,
                                                             shuffle_channel_id)

        hidden_query, label_new_query, label_new_support, lam = self.mixup_data(hidden_support, label_new_support, hidden_query,
                                                        label_new_query)

        return hidden_support, label_new_support, hidden_query, label_new_query, lam

    def functional_forward_cf_mix_query(self, hidden_support, label_support, hidden_query, label_query, sel_layer,
                                                 shuffle_list, shuffle_channel_id, weights,
                                                 is_training=True):

        flag = 0

        for layer in range(4):
            if layer == sel_layer:
                hidden_support, label_new_support, hidden_query, label_new_query, lam = self.mix_cf(hidden_support,
                                                                                                    label_support,
                                                                                                    hidden_query,
                                                                                                    label_query,
                                                                                                    shuffle_list,
                                                                                                    shuffle_channel_id)
                flag = 1

            if not flag:
                hidden_support = self.functional_conv_block(hidden_support, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                            weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                            is_training)

            hidden_query = self.functional_conv_block(hidden_query, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                      weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                      is_training)


        hidden_query = hidden_query.view(hidden_query.size(0), -1)

        x = F.linear(hidden_query, weights['logits.weight'], weights['logits.bias'])

        return x, label_new_query, label_new_support, lam
