import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import euclidean_dist
from learner import FCNet
import numpy as np
from torch.distributions import Beta
from collections import OrderedDict
import random

class MAML(nn.Module):
    def __init__(self, args):
        super(MAML, self).__init__()
        self.args = args
        self.learner = FCNet(args=args, x_dim=2866, hid_dim=64, dropout=0.2)
        self.dist = Beta(torch.FloatTensor([3]), torch.FloatTensor([5]))
        self.loss_fn = nn.CrossEntropyLoss()
        if args.train:
            self.num_updates = self.args.num_updates
        else:
            self.num_updates = self.args.num_updates_test
    def grad_similarity(self,xs,ys,xq,yq):
        fast_weights = OrderedDict(self.learner.named_parameters())
        logits = self.learner.functional_forward(xs, fast_weights, is_training=True)
        loss = self.loss_fn(logits, ys)
        gradients = torch.autograd.grad(loss, fast_weights.values())  

        gradients1 = []
        for i,item in enumerate(gradients):
            if i == 0:
                gradients1 = gradients[i].reshape(-1)
            else:
                gradients1 = torch.cat((gradients1,gradients[i].reshape(-1)))
        # print('gradients1',gradients1)
        fast_weights = OrderedDict(self.learner.named_parameters())        
        query_logits = self.learner.functional_forward(xq, fast_weights)
        query_loss = self.loss_fn(query_logits, yq)        
        gradients = torch.autograd.grad(query_loss, fast_weights.values())   
        gradients2 = []
        for i,item in enumerate(gradients):
            if i == 0:
                gradients2 = gradients[i].reshape(-1)
            else:
                gradients2 = torch.cat((gradients2,gradients[i].reshape(-1)))
        AD_loss = torch.cosine_similarity(gradients1,gradients2,dim=0)
        return AD_loss
    
    def forward_query_loss(self, xs, ys, xq, yq):
        create_graph = True
        
        fast_weights = OrderedDict(self.learner.named_parameters())
        # for (name,param) in fast_weights.items():
        #     print(name)
        # print('fast_weights',self.learner.named_parameters())
        for inner_batch in range(5):
            logits = self.learner.functional_forward(xs, fast_weights, is_training=True)
            loss = self.loss_fn(logits, ys)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)
            
            fast_weights = OrderedDict(
                (name, param - self.args.update_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )
    

        query_logits = self.learner.functional_forward(xq, fast_weights)
        query_loss = self.loss_fn(query_logits, yq)

        y_pred = query_logits.softmax(dim=1).max(dim=1)[1]
        query_acc = (y_pred == yq).sum().float() / yq.shape[0]

        return query_loss, query_acc
    
    
    def forward(self, xs, ys, xq, yq):
        create_graph = True
        fast_weights = OrderedDict(self.learner.named_parameters())
        
        for inner_batch in range(self.num_updates):
            logits = self.learner.functional_forward(xs, fast_weights, is_training=True)
            loss = self.loss_fn(logits,ys)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph = create_graph)
            
            fast_weights = OrderedDict(
                (name, param - self.args.update_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )
        query_logits = self.learner.functional_forward(xq, fast_weights)
        query_loss = self.loss_fn(query_logits, yq)
        y_pred = query_logits.softmax(dim=1).max(dim=1)[1]
        query_acc = (y_pred == yq).sum().float() / yq.shape[0]
        return query_loss, query_acc
            
    
    def forward_ours(self, xs, ys, xs_Aug, xq, xq_Aug, yq):
        create_graph = True
        fast_weights = OrderedDict(self.learner.named_parameters())
        sel_loop = 3
        global sel_layer
        for inner_batch in range(self.num_updates):
            if inner_batch <= sel_loop:
                logits = self.learner.functional_forward(xs, fast_weights, is_training=True)
                loss = self.loss_fn(logits, ys)
                gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)
                fast_weights = OrderedDict(
                    (name, param - self.args.update_lr * grad)
                    for ((name, param), grad) in zip(fast_weights.items(), gradients)
                )
            else:    
                sel_layer = 1 # random.randint(0,1)
                logits = self.learner.forward_gen_ours(xs, xs_Aug, fast_weights, sel_layer, is_training=True)
                loss = self.loss_fn(logits, ys)
                gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)
                fast_weights = OrderedDict(
                    (name, param - self.args.update_lr * grad)
                    for ((name, param), grad) in zip(fast_weights.items(), gradients)
                )
        query_logits = self.learner.forward_gen_ours(xq, xq_Aug, fast_weights, sel_layer, is_training=True)
        query_loss = self.loss_fn(query_logits, yq)
        y_pred = query_logits.softmax(dim=1).max(dim=1)[1]
        query_acc = (y_pred == yq).sum().float() / yq.shape[0]
        return query_loss, query_acc


    def forward_metamix(self, xs, ys, xq, yq):
        create_graph = True
        fast_weights = OrderedDict(self.learner.named_parameters())
        for inner_batch in range(self.num_updates):
                logits = self.learner.functional_forward(xs, fast_weights, is_training=True)
                loss = self.loss_fn(logits, ys)
                gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)
                fast_weights = OrderedDict(
                    (name, param - self.args.update_lr * grad)
                    for ((name, param), grad) in zip(fast_weights.items(), gradients)
                )
        query_logits, reweighted_yq, reweighted_ys, lam = self.learner.forward_metamix(xq, yq, xq, yq, fast_weights)
        
        
        
        query_loss = lam*self.loss_fn(query_logits, reweighted_yq) + (1-lam)*self.loss_fn(query_logits, reweighted_ys)
        y_pred = query_logits.softmax(dim=1).max(dim=1)[1]
        query_acc = (y_pred == yq).sum().float() / yq.shape[0]
        return query_loss, query_acc         


    def forward_within(self, xs, ys, xq, yq):
        lam_mix = self.dist.sample().to("cuda")

        z = self.learner(xs)

        z_dim = z.size(-1)

        z_proto = z.view(self.args.num_classes, self.args.update_batch_size, z_dim).mean(1)

        zq, reweighted_yq, reweighted_ys, lam = self.learner.forward_within(xq, yq, xq, yq, lam_mix)

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1)

        loss_val = []
        for i in range(self.args.num_classes * self.args.update_batch_size_eval):
            loss_val.append(-log_p_y[i, reweighted_yq[i]]*lam -log_p_y[i, reweighted_ys[i]] * (1-lam))

        loss_val = torch.stack(loss_val).squeeze().mean()

        # accuracy evaluation
        zq_real = self.learner(xq)

        dists_real = euclidean_dist(zq_real, z_proto)

        log_p_y_real = F.log_softmax(-dists_real, dim=1)

        _, y_hat = log_p_y_real.max(1)

        acc_val = torch.eq(y_hat, yq).float().mean()

        return loss_val, acc_val

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam.cpu())
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def mixup_data(self, xs, xq, lam):
        mixed_x = lam * xq + (1 - lam) * xs

        return mixed_x, lam


    def forward_crossmix(self, x1s, y1s, x1q, y1q, x2s, y2s, x2q, y2q):
        lam_mix = self.dist.sample().to("cuda")

        task_2_shuffle_id = np.arange(self.args.num_classes)
        np.random.shuffle(task_2_shuffle_id)
        task_2_shuffle_id_s = np.array(
            [np.arange(self.args.update_batch_size) + task_2_shuffle_id[idx] * self.args.update_batch_size for idx in
             range(self.args.num_classes)]).flatten()
        task_2_shuffle_id_q = np.array(
            [np.arange(self.args.update_batch_size_eval) + task_2_shuffle_id[idx] * self.args.update_batch_size_eval for
             idx in range(self.args.num_classes)]).flatten()

        x2s = x2s[task_2_shuffle_id_s]
        x2q = x2q[task_2_shuffle_id_q]

        x_mix_s, _ = self.mixup_data(self.learner.net[0](x1s), self.learner.net[0](x2s), lam_mix)

        x_mix_q, _ = self.mixup_data(self.learner.net[0](x1q), self.learner.net[0](x2q), lam_mix)

        x = torch.cat([x_mix_s, x_mix_q], 0)

        z = self.learner.forward_crossmix(x)

        z_dim = z.size(-1)

        z_proto = z[:self.args.num_classes * self.args.update_batch_size].view(self.args.num_classes,
                                                                               self.args.update_batch_size, z_dim).mean(
            1)
        zq = z[self.args.num_classes * self.args.update_batch_size:]

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1)

        loss_val = []
        for i in range(self.args.num_classes * self.args.update_batch_size_eval):
            loss_val.append(-log_p_y[i, y1q[i]])

        loss_val = torch.stack(loss_val).squeeze().mean()

        _, y_hat = log_p_y.max(1)

        acc_val = torch.eq(y_hat, y1q).float().mean()

        return loss_val, acc_val
