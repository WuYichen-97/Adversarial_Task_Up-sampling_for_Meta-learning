import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.zdim = 1024
        # self.fc1 = nn.Linear(2866,2866)
        # self.fc2 = nn.Linear(2866,2866)
        
        self.conv1 = nn.Conv1d(1,32,1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32,64,1)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x0, x1):
        # x0 = x0.unsqueeze(1)
        # x1 = x1.unsqueeze(1)

        
        x = F.relu(self.bn1(self.conv1(x0)))    
        x = F.relu(self.bn2(self.conv2(x)))    
        # print('x',x.shape)         

        v = x.reshape(-1,5,64,2866)
        v = torch.mean(v,dim=0) #(5,128,84,84)
        # print('v',v.shape)

        

        x1 = x1.unsqueeze(0)- x0.unsqueeze(1)
        x1 = x1.reshape(-1,1,2866)
        # print('x1',x1.shape)
        
        x1 = F.relu(self.bn1(self.conv1(x1)))    
        x1 = F.relu(self.bn2(self.conv2(x1)))       

        res_v = x1.reshape(int(x1.shape[0]/15),5,3,64,2866) #.permute(1,0,2,3,4,5) #[5,8,3,128,84,84]
        res_v = torch.mean(res_v,dim=0).reshape(15,64,2866) #(15,32,2866)
        # print('res_v',res_v.shape)
        return x,x1,v,res_v        
    
        

class Decoder(nn.Module):
    def __init__(self, num_coarse=1024, num_dense=16384):
        super(Decoder, self).__init__()

        self.num_coarse = num_coarse
        self.conv1 = nn.Conv1d(129, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)  
        self.conv2 = nn.Conv1d(64, 32, 1)
        self.bn2 = nn.BatchNorm1d(32)        
        self.conv3 = nn.Conv1d(32, 1, 1)
        # self.conv1 = nn.ConvTranspose2d(129, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)  
        # self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(32)        
        # self.conv3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(32)   
        # self.conv4 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1, bias=False)   
        


        self.conv11 = nn.Conv1d(128, 64, 1)
        self.bn11 = nn.BatchNorm1d(64) 
        self.conv21= nn.Conv1d(64, 32, 1)
        self.bn21 = nn.BatchNorm1d(32)     
        self.conv31 = nn.Conv1d(32, 16, 1)
        # self.bn31 = nn.BatchNorm2d(3)  
        self.fc1 = nn.Linear(16*2866, 256) 
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,1)
          
        
    
    def forward(self, x, x1, x0, v, res_v):
        # x0 = x0.unsqueeze(1)
        
        x = x.repeat_interleave(3,dim=0)
        # print('x',x.shape)
        x = x.repeat(2,1,1)
        x1 = x1.repeat(2,1,1)
        # print('x',x.shape)
        # print('x1',x1.shape)
        
        noise1 = torch.tensor([-0.5]).repeat(300,1,2866)
        noise2 = torch.tensor([0.5]).repeat(300,1,2866)
        noise = torch.cat([noise1,noise2],dim=0).cuda()
        # print('noise',noise.shape)
        
        x1 = torch.cat([x,x1,noise],dim=1) 
        # print('x1',x1.shape)
        
        
        x_res = self.bn1(self.conv1(x1))
        x_res = self.bn2(self.conv2(x_res))   
        # x_res = self.bn3(self.conv3(x_res))
        x_res = self.conv3(x_res)
        x_res = x_res.reshape(-1,3,1,2866)
        # print('x_res', x_res.shape)

        v = v.repeat_interleave(3,dim=0)
        v = torch.cat([v,res_v],dim=1)

        s1 = self.bn11(self.conv11(v))
        s1 = self.bn21(self.conv21(s1)) 
        s1 = self.conv31(s1)
        
        s1 = self.fc1(s1.reshape(s1.shape[0],-1))
        s1 = self.fc2(s1)
        s1 = self.fc3(s1)
        s1 = s1.reshape(5,3)
        s2 = s1.repeat(int(x0.shape[0]/5*2),1)
        s2 = F.softmax(20*s2,dim=1).reshape(-1,3,1,1)
        x_res = (s2*x_res).sum(dim=1)
        x0 = x0.repeat(2,1,1)
        # print('x0',x0.shape)
        x = x_res + x0
        # x = F.sigmoid(x_res+x0)    
        
        y_coarse = x.reshape(x.shape[0],2866)
        return y_coarse, torch.argmax(s1,dim=1)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x0,x1):
        x,x1,v,res_v = self.encoder(x0,x1)
        y_coarse,s1 = self.decoder(x,x1,x0,v,res_v)
        return  x, y_coarse,s1

if __name__ == "__main__":
    pcs = torch.rand(100,1,2866)#.cuda()
    pcs1 = torch.rand(3,1,2866)#.cuda()
    # lam = torch.tensor([0.7])
    # pcs2 = torch.rand(40,3,84,84)

    ae = AutoEncoder()#.cuda()
    x,y_coarse,s1= ae(pcs,pcs1)

    print( y_coarse.size())