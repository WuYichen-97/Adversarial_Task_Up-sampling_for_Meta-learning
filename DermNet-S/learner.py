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


    # def mixup_data(self, xs, xs2, lam=None):
    #     if not lam:
    #         lam = self.dist.sample().cuda()
    #     mixed_xs = lam * xs + (1 - lam) * xs2
    #     return mixed_xs, lam
    
    # def mixup_data1(self, xs, xs2, lam=None):
    #     if not lam:
    #         lam = 0
    #     mixed_xs = lam * xs + (1 - lam) * xs2
    #     return mixed_xs, lam    



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

    def mixup_data(self, xs, ys, xq, yq, lam):
        query_size = xq.shape[0]
        shuffled_index = torch.randperm(query_size)
        xs = xs[shuffled_index]
        ys = ys[shuffled_index]
        mixed_x = lam * xq + (1 - lam) * xs
        return mixed_x, yq, ys, lam


    def forward_metamix(self, hidden_support, label_support, hidden_query, label_query, weights, is_training=True):
        lam_mix = self.dist.sample().cuda()
        sel_layer = random.randint(0, 3)
        flag = 0
        for layer in range(4):
            if layer==sel_layer:
                hidden_support, reweighted_query, reweighted_support, lam= self.mixup_data(hidden_support, label_support, hidden_query, label_query, lam_mix)
                flag=1
            if not flag:
                hidden_query = self.functional_conv_block(hidden_query, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                is_training)     
            hidden_support = self.functional_conv_block(hidden_support, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                            weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                            is_training)     
                
        mix_supp = hidden_support.view(hidden_support.size(0), -1)
        logits  = F.linear(mix_supp , weights['logits.weight'], weights['logits.bias'])
        return logits, reweighted_query, reweighted_support, lam




 
    
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



