import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
# from model_conv9LN_res import AutoEncoder
import random
# from layers_BCE import SinkhornDistance
from layers import SinkhornDistance
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(0)
np.random.seed(1)

class TaskGenerator():
   def __init__(self, AutoEncoder, n_shot):
        self.n_shot = n_shot
        self.network = AutoEncoder()
        if self.n_shot  == 1 :
            self.network.load_state_dict(torch.load('./ATU'))
        else:
            self.network.load_state_dict(torch.load('./ATU5'))
        self.optimizer_network = torch.optim.Adam(self.network.parameters(), lr=1e-5, weight_decay=1e-6)
        self.DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.sinkhorn = SinkhornDistance(eps=0.01, max_iter=500, reduction='mean')
    
   def Upsampling(self, maml, ys, yq, partial_input, gt, MB_feats, step):
       '''
       task_ori_pool (16,5,32,42,42)  (20,5,32,42,42)
       task_mix_pool (16,5,32,42,42)  (20,5,32,42,42)
       '''    
       self.network.train().cuda()
       
       if self.n_shot==1:
           _, preds, _ = self.network(partial_input, MB_feats)
           # out = preds.reshape(40,5,3,84,84)
           out = gt.reshape(32,5,3,84,84)
           task1 = out[:16]
           task2 = out[16:]
           
           # supp1, query1 = task1[:].permute(1,0,2,3,4).reshape(25,3,84,84),task1[5:].permute(1,0,2,3,4).reshape(75,3,84,84)       
           # supp2, query2 = task2[:5].permute(1,0,2,3,4).reshape(25,3,84,84),task2[5:].permute(1,0,2,3,4).reshape(75,3,84,84)     
    
           supp1, query1 = task1[0],task1[1:].permute(1,0,2,3,4).reshape(75,3,84,84)       
           supp2, query2 = task2[0],task2[1:].permute(1,0,2,3,4).reshape(75,3,84,84)  
    
           loss_val1, acc_val = maml.forward_query_loss(supp1,ys,query1,yq)
           loss_val2, acc_val = maml.forward_query_loss(supp2,ys,query2,yq)
           
           loss_grad1 = maml.grad_similarity(supp1,ys,query1,yq)
           loss_grad2 = maml.grad_similarity(supp2,ys,query2,yq)
    
           loss,_,_ = self.sinkhorn(preds.reshape(2,80,-1), gt.reshape(2,80,-1))
           loss = loss + 0.5*loss_grad1 + 0.5*loss_grad2 - 0.5* loss_val1 - 0.5*loss_val2
       else:
           _, preds, _ = self.network(partial_input, MB_feats)
           out = preds.reshape(40,5,3,84,84)
           # out = gt.reshape(32,5,3,84,84)
           task1 = out[:20]
           task2 = out[20:]
           
           supp1, query1 = task1[:5].permute(1,0,2,3,4).reshape(25,3,84,84),task1[5:].permute(1,0,2,3,4).reshape(75,3,84,84)       
           supp2, query2 = task2[:5].permute(1,0,2,3,4).reshape(25,3,84,84),task2[5:].permute(1,0,2,3,4).reshape(75,3,84,84)     
    
           # supp1, query1 = task1[0],task1[1:].permute(1,0,2,3,4).reshape(75,3,84,84)       
           # supp2, query2 = task2[0],task2[1:].permute(1,0,2,3,4).reshape(75,3,84,84)  
    
           loss_val1, acc_val = maml.forward_query_loss(supp1,ys,query1,yq)
           loss_val2, acc_val = maml.forward_query_loss(supp2,ys,query2,yq)
           
           loss_grad1 = maml.grad_similarity(supp1,ys,query1,yq)
           loss_grad2 = maml.grad_similarity(supp2,ys,query2,yq)
    
           loss,_,_ = self.sinkhorn(preds.reshape(2,100,-1), gt.reshape(2,100,-1))
           loss = loss + 0.5*loss_grad1 + 0.5*loss_grad2 - 0.5* loss_val1 - 0.5*loss_val2           
           
           
       self.optimizer_network.zero_grad()   
       loss.backward()
       self.optimizer_network.step() 
       
       
       
       if step% 1000 == 0:
              print('loss',step,loss)#,loss_grad1,loss_grad2,loss_val1,loss_val2)#,-20*loss_val,-10*loss_grad)#,50*s)
       if np.random.rand() < 0.5:
           return supp1.detach(), query1.detach()#, task1.detach()
       else:
           return supp2.detach(), query2.detach()#, task2.detach()


       
      
       
       
   
    
