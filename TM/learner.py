import torch
import torch.nn as nn
from torch.distributions import Beta
import torch.nn.functional as F

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class FCNet(nn.Module):
    def __init__(self, args, x_dim, hid_dim, dropout=0.2):
        super(FCNet, self).__init__()
        self.args = args
        self.drop_ratio = dropout
        self.net = nn.Sequential(
            self.fc_block(x_dim, hid_dim, dropout),
            self.fc_block(hid_dim, hid_dim, dropout),
        )
        self.dist = Beta(torch.FloatTensor([5]), torch.FloatTensor([3]))
        # self.dist1 = Beta(torch.FloatTensor([5]), torch.FloatTensor([3]))
        
        self.hid_dim = hid_dim
        self.logits = nn.Linear(64, self.args.num_classes)
        

    def fc_block(self, in_features, out_features, dropout):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
    
    def functional_fc_block(self, x, weights, biases, bn_weights, bn_biases, dropout, is_training=True):
        x = F.linear(x,weights,biases)
        x = F.batch_norm(x,running_mean=None,running_var=None, weight=bn_weights, bias=bn_biases, training=is_training)
        x = F.relu(x)
        x = F.dropout(x,p=dropout,training=is_training)
        return x
    
    
    def functional_forward(self,x,weights,is_training=True):
        for block in range(2):
            x = self.functional_fc_block(x, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'], 
                                         weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'), self.drop_ratio, is_training)
        x = x.view(x.size(0),-1)
        x = F.linear(x,weights['logits.weight'],weights['logits.bias'])
        return x
        
    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0),-1)
        return self.logits(x) 


    def forward_gen_ours(self, hidden_support, hidden_support2, weights, sel_layer, is_training=True):
        
        flag = 0
        for layer in range(2):
            if layer==sel_layer:
                hidden_support = hidden_support2
                # hidden_support,lam,_ = self.mixup_data_ours(hidden_support, hidden_support2, index, lamda)
                flag=1
            if not flag:
                hidden_support2 = self.functional_fc_block(hidden_support2, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),self.drop_ratio,
                                                is_training)     
            hidden_support = self.functional_fc_block(hidden_support, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                            weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),self.drop_ratio,
                                            is_training)                     
        hidden_support = hidden_support.view(hidden_support.size(0), -1)
        logits  = F.linear(hidden_support , weights['logits.weight'], weights['logits.bias'])
        
        return logits

    def COindex_ours(self, T1, T2):
        T1 = T1.reshape(len(T1),-1).detach()
        T2 = T2.reshape(len(T2),-1).detach()
        dist = torch.cdist(T1.unsqueeze(0), T2.unsqueeze(0)).squeeze(0)
        for i in range(len(T1)):
            d = dist[i]
            _, index = torch.min(d, 0)
            if i == 0:
                out_index = index.unsqueeze(0)#.unsqueeze(0)
            else:
                out_index = torch.cat((out_index, index.unsqueeze(0)), 0)
        return out_index


    def mixup_data_ours(self, xs, xs2, index = None, lam=None):
        set_size = xs.shape[0]
        xs_mean = xs.clone()
        xs_mean = xs_mean.reshape(5,int(set_size/5),xs_mean.shape[1])
        xs_mean = torch.mean(xs_mean, dim=1)
        
        xs2 = xs2.reshape(3,1,xs2.shape[1]) #[:3,0].unsqueeze(1)
        if index is None:
            xs2_mean = torch.mean(xs2,dim=1)
            index = self.COindex_ours(xs_mean, xs2_mean)
        xs = xs.reshape(5,int(set_size/5),xs.shape[1])    
        xs2 = xs2[index.long()]

        'Each class have the same lam for both supp & query'
        if not lam:      
            lam = self.dist.sample().cuda()
        mix_feature = torch.zeros_like(xs)
        idx = 0
        for i in range(5):
            for j in range(int(set_size/5)):
                lam_i = lam
                mix_feature[i,j] = lam_i*xs[i,j]+ (1-lam_i)*xs2[i,idx]
        mix_feature = mix_feature.reshape(set_size,xs.shape[2])    
        return mix_feature, lam, index

    
    def forward_ours_supp(self, inp_support, inp_query, weights):
        sel_layer = 1
        block = 0
        hidden1_support = self.functional_fc_block(inp_support, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'], 
                                         weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'), self.drop_ratio)
        hidden1_query = self.functional_fc_block(inp_query, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'], 
                                         weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'), self.drop_ratio)        
        
        if sel_layer == 1:
            hidden1_query, lam, index = self.mixup_data_ours(hidden1_support, hidden1_query)
        
        block = 1
        hidden2_query = self.functional_fc_block(hidden1_query, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'], 
                                         weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'), self.drop_ratio)              
        # hidden2_query = self.net[1](hidden1_query)
        hidden3_query = hidden2_query.view(hidden2_query.size(0),-1)
        logits  = F.linear(hidden3_query , weights['logits.weight'], weights['logits.bias'])
        return logits, lam, index
        
    def forward_ours_query(self, inp_support,  inp_query, weights, lam, index):
        block = 0
        hidden1_support = self.functional_fc_block(inp_support, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'], 
                                         weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'), self.drop_ratio)
        hidden1_query = self.functional_fc_block(inp_query, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'], 
                                         weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'), self.drop_ratio)  
        hidden1_query, lam, index = self.mixup_data_ours(hidden1_support, hidden1_query, index, lam)
                
        
        
        block = 1
        hidden2_query = self.functional_fc_block(hidden1_query, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'], 
                                         weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'), self.drop_ratio)   
        hidden3_query = hidden2_query.view(hidden2_query.size(0),-1)
        logits  = F.linear(hidden3_query , weights['logits.weight'], weights['logits.bias'])
        return logits

    def forward_ours_query1(self, inp_support, inp_query, weights):
        
        lam = self.dist.sample().cuda()
        block = 0
        hidden1_support = self.functional_fc_block(inp_support, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'], 
                                         weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'), self.drop_ratio)
        hidden1_query = self.functional_fc_block(inp_query, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'], 
                                         weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'), self.drop_ratio)  
        hidden1_query, lam, index = self.mixup_data_ours(hidden1_support, hidden1_query, lam=lam) 
        
        
        block = 1
        hidden2_query = self.functional_fc_block(hidden1_query, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'], 
                                         weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'), self.drop_ratio)   
        hidden3_query = hidden2_query.view(hidden2_query.size(0),-1)
        logits  = F.linear(hidden3_query , weights['logits.weight'], weights['logits.bias'])
        return logits,lam,index








    def mixup_data(self, xs, ys, xq, yq, lam):
        query_size = xq.shape[0]
        shuffled_index = torch.randperm(query_size)
        xs = xs[shuffled_index]
        ys = ys[shuffled_index]
        mixed_x = lam * xq + (1 - lam) * xs
        return mixed_x, yq, ys, lam


    def forward_metamix(self, inp_support, label_support, inp_query, label_query, weights):
        lam_mix = self.dist.sample().cuda()
        block = 0
        hidden1_support = self.functional_fc_block(inp_support, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'], 
                                         weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'), self.drop_ratio)
        hidden1_query = self.functional_fc_block(inp_query, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'], 
                                         weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'), self.drop_ratio)  
        hidden1_query, reweighted_query, reweighted_support, lam = self.mixup_data(hidden1_support, label_support, hidden1_query,
                                                               label_query, lam_mix)    
        block = 1
        hidden2_query = self.functional_fc_block(hidden1_query, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'], 
                                         weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'), self.drop_ratio)   
        hidden3_query = hidden2_query.view(hidden2_query.size(0),-1)
        logits  = F.linear(hidden3_query , weights['logits.weight'], weights['logits.bias'])
        return logits, reweighted_query, reweighted_support, lam


    def forward_crossmix(self, x):
        return self.net[1](x)


    
    def forward_within(self, inp_support, label_support, inp_query, label_query, lam_mix):
        sel_layer = 1
        reweighted_query = label_query
        reweighted_support = label_support
        hidden1_support = self.net[0](inp_support)
        hidden1_query = self.net[0](inp_query)
        if sel_layer == 1:
            hidden1_query, reweighted_query, reweighted_support, lam = self.mixup_data(hidden1_support, label_support, hidden1_query,
                                                               label_query, lam_mix)
        hidden2_query = self.net[1](hidden1_query)
        return hidden2_query, reweighted_query, reweighted_support, lam

















