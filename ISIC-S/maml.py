import torch
import torch.nn as nn
from collections import OrderedDict
from learner import Conv_Standard
import numpy as np
import ipdb
import torch.nn.functional as F
import random





def get_inner_loop_parameter_dict(params):
    """
    Returns a dictionary with the parameters to use for inner loop updates.
    :param params: A dictionary of the network's parameters.
    :return: A dictionary of the parameters to use for the inner loop optimization process.
    """
    param_dict = dict()
    for name, param in params.items():
            if "net.0" not in name :
                param_dict[name] = param
    return param_dict

def get_fixed_parameter_dict(params):
    """
    Returns a dictionary with the parameters to use for inner loop updates.
    :param params: A dictionary of the network's parameters.
    :return: A dictionary of the parameters to use for the inner loop optimization process.
    """
    param_dict = dict()
    for name, param in params.items():
            if "net.0" in name :
                param_dict[name] = param
    return param_dict



class MAML(nn.Module):
    def __init__(self, args):
        super(MAML, self).__init__()
        self.args = args
        self.learner = Conv_Standard(args=args, x_dim=3, hid_dim=args.num_filters, z_dim=args.num_filters,
                                     final_layer_size=800)
        self.loss_fn = nn.CrossEntropyLoss()
        if self.args.train:
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
        # for (name,param) in fast_weights.items():
        #     print(name)
        # print('fast_weights',self.learner.named_parameters())
        for inner_batch in range(self.num_updates):
            logits = self.learner.functional_forward(xs, fast_weights, is_training=True)
            loss = self.loss_fn(logits, ys)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)
            
            fast_weights = OrderedDict(
                (name, param - self.args.update_lr * grad)
                if 'net.3' in name or 'logits' in name else (name, param) for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )
    
            

        query_logits = self.learner.functional_forward(xq, fast_weights)
        query_loss = self.loss_fn(query_logits, yq)

        y_pred = query_logits.softmax(dim=1).max(dim=1)[1]
        query_acc = (y_pred == yq).sum().float() / yq.shape[0]

        return query_loss, query_acc


    def forward_yao(self, xs, ys, xq, yq, xs2, xq2):
        create_graph = True

        fast_weights = OrderedDict(self.learner.named_parameters())
        
        for inner_batch in range(self.num_updates-1):
            logits = self.learner.functional_forward(xs, fast_weights, is_training=True)
            loss = self.loss_fn(logits, ys)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)
            
            fast_weights = OrderedDict(
                (name, param - self.args.update_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )

        if self.args.train and self.args.mix:

            logits, sel_layer, lam = self.learner.forward_metamix_supp(xs, xs2, fast_weights)
            loss = self.loss_fn(logits, ys)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)
            fast_weights = OrderedDict(
                (name, param - self.args.update_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )            

            query_logits = self.learner.forward_metamix_query(xq, xq2, fast_weights, lam, sel_layer)
            query_loss = self.loss_fn(query_logits, yq)
            y_pred = query_logits.softmax(dim=1).max(dim=1)[1]
            query_acc = (y_pred == yq).sum().float() / yq.shape[0]            
            
        else:
            logits = self.learner.functional_forward(xs, fast_weights, is_training=True)
            loss = self.loss_fn(logits, ys)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)
            
            fast_weights = OrderedDict(
                (name, param - self.args.update_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )
            
            query_logits = self.learner.functional_forward(xq, fast_weights, is_training=False)
            query_loss = self.loss_fn(query_logits, yq)

            y_pred = query_logits.softmax(dim=1).max(dim=1)[1]
            query_acc = (y_pred == yq).sum().float() / yq.shape[0]
        return query_loss, query_acc

    def forward_yao_same_task(self, xs, ys, xq, yq):
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

        if self.args.train and self.args.mix:
            
            'Random sample query sample to keep the same with support set, for 1-shot this code'
            idx = np.random.randint(0,15)
            xq = xq.reshape(5,15,3,84,84)
            yq = yq.reshape(5,15)
            xq = xq[:,idx,:,:,:]
            yq = yq[:,idx]
            
            
            query_logits, query_label_new, support_label_new, lam = self.learner.forward_metamix(xs, ys, xq, yq,
                                                                                                 fast_weights)
            query_loss = lam * self.loss_fn(query_logits, query_label_new) + (1 - lam) * self.loss_fn(query_logits,
                                                                                                      support_label_new)

            y_pred = query_logits.softmax(dim=1).max(dim=1)[1]
            query_acc = (y_pred == query_label_new).sum().float() / query_label_new.shape[0]
        else:
            query_logits = self.learner.functional_forward(xq, fast_weights)
            query_loss = self.loss_fn(query_logits, yq)

            y_pred = query_logits.softmax(dim=1).max(dim=1)[1]
            query_acc = (y_pred == yq).sum().float() / yq.shape[0]

        return query_loss, query_acc

    def forward_ours1(self, xs, ys, xq, yq, xs1, xq1):
        create_graph = True

        fast_weights = OrderedDict(self.learner.named_parameters())

        for inner_batch in range(self.num_updates-1):
            logits = self.learner.functional_forward(xs, fast_weights, is_training=True)
            loss = self.loss_fn(logits, ys)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)
            
            fast_weights = OrderedDict(
                (name, param - self.args.update_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )
        

        if self.args.train and self.args.mix:
            logits, sel_layer, lam = self.learner.forward_metamix_supp1(xs, xs1, fast_weights)
            loss = self.loss_fn(logits, ys)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)
            
            fast_weights = OrderedDict(
                (name, param - self.args.update_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )            

            query_logits = self.learner.forward_metamix_query1(xq, xq1, fast_weights, lam, sel_layer)
            query_loss = self.loss_fn(query_logits, yq)
            y_pred = query_logits.softmax(dim=1).max(dim=1)[1]
            query_acc = (y_pred == yq).sum().float() / yq.shape[0]
            
        else:
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
    
    
    def forward_ours(self, xs, ys, xs_Aug, xq, xq_Aug, yq):
        create_graph = True

        fast_weights = OrderedDict(self.learner.named_parameters())
        
        # sel_layer = random.randint(0,3)
        sel_loop = 3
        global sel_layer
        for inner_batch in range(self.num_updates):
            if inner_batch <= sel_loop:
                logits = self.learner.functional_forward(xs, fast_weights, is_training=True)
                loss = self.loss_fn(logits, ys)
                gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)
                
                fast_weights = OrderedDict(
                    (name, param - self.args.update_lr * grad)
                    if 'net.3' in name or 'logits' in name else (name, param) for ((name, param), grad) in zip(fast_weights.items(), gradients)
                )
            else:    
                sel_layer =  random.randint(0, 2)
                logits = self.learner.forward_gen_ours(xs, xs_Aug, fast_weights, sel_layer, is_training=True)
                loss = self.loss_fn(logits, ys)
                gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)
                
                fast_weights = OrderedDict(
                    (name, param - self.args.update_lr * grad)
                    if 'net.3' in name or 'logits' in name else (name, param) for ((name, param), grad) in zip(fast_weights.items(), gradients)
                )
        
        # query_logits = self.learner.functional_forward(xq, fast_weights)
        query_logits = self.learner.forward_gen_ours(xq, xq_Aug, fast_weights, sel_layer, is_training=True)
        query_loss = self.loss_fn(query_logits, yq)
        y_pred = query_logits.softmax(dim=1).max(dim=1)[1]
        query_acc = (y_pred == yq).sum().float() / yq.shape[0]

        return query_loss, query_acc
    
    
    
    def forward_ours_features(self, xs, ys, xq, yq, MB_X, sel_layer):
        create_graph = True

        fast_weights = OrderedDict(self.learner.named_parameters())
        # print('fast_weights',fast_weights) 
        
        for inner_batch in range(self.num_updates-1):
            logits = self.learner.functional_forward(xs, fast_weights, is_training=True)
            loss = self.loss_fn(logits, ys)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)
            
            fast_weights = OrderedDict(
                (name, param - self.args.update_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )
            

        # PCN task Aug Part
        feats_supp = self.learner.functional_forward_features(xs, sel_layer, fast_weights, is_training=True)
        feats_MB = self.learner.functional_forward_features(MB_X, sel_layer, fast_weights)
        feats_query = self.learner.functional_forward_features(xq, sel_layer, fast_weights)
        return feats_supp.detach(), ys, feats_MB.detach(), feats_query.detach(), yq, fast_weights

    def forward_ours_features1(self, xs, ys, xq, yq, sel_layer):
        create_graph = True

        fast_weights = OrderedDict(self.learner.named_parameters())
        # fixed_weights = get_fixed_parameter_dict(fast_weights)
        # update_weights = get_inner_loop_parameter_dict(fast_weights)
        for inner_batch in range(self.num_updates-1):
            logits = self.learner.functional_forward(xs, fast_weights, is_training=True)
            loss = self.loss_fn(logits, ys)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)
            
            fast_weights = OrderedDict(
                (name, param - self.args.update_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )
        # PCN task Aug Part
        feats_supp = self.learner.functional_forward_features(xs, sel_layer, fast_weights, is_training=True)
        feats_query = self.learner.functional_forward_features(xq, sel_layer, fast_weights)
        return feats_supp.detach(), ys, feats_query.detach(), yq

    def forward_ours_features_loss(self, supp, ys, query, yq, sel_layer):
        create_graph = True
        fast_weights = OrderedDict(self.learner.named_parameters())
        # fixed_weights = get_fixed_parameter_dict(fast_weights)
        # fast_weights = get_inner_loop_parameter_dict(fast_weights)

        logits = self.learner.functional_forward_classifier(supp, fast_weights, sel_layer, is_training=True)
        loss = self.loss_fn(logits, ys)
        gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph, allow_unused=True)
        fast_weights = OrderedDict(
            (name, param - self.args.update_lr * grad) if grad is not None else (name, param)
            for ((name, param), grad) in zip(fast_weights.items(), gradients))

        query_logits = self.learner.functional_forward_classifier(query, fast_weights, sel_layer, is_training=True)
        query_loss = self.loss_fn(query_logits, yq)

        y_pred = query_logits.softmax(dim=1).max(dim=1)[1]
        query_acc = (y_pred == yq).sum().float() / yq.shape[0]
        return query_loss, query_acc, fast_weights
