import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import csv # for splitting
import random

class ISIC(data.Dataset):
    """
    put mini_imagenet as:
    | mini-imagenet
        | images
            | file1.jpg
            | file2.jpg 
            .
            .   
    | train.csv
    | val.csv
    | test.csv
    In meta-learning:
    batch = task
    set = n_way * k_shot for per train task(batch), n_way * k_query for per test task(batch)
    """
    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, startidx=0):
        """
        n_way : number of classes
        k_shot : number of pictures in each class
        5 way 1 shot
        """
        self.batchsz = batchsz
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query # for evaluation
        self.setsz = self.n_way * self.k_shot
        self.querysz = self.n_way * self.k_query
        self.resize = resize
        self.startidx = startidx
        print('shuffle DB: %s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
            mode, batchsz, n_way, k_shot, k_query, resize
        ))

        if mode == 'train':
            self.transform = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.Resize((self.resize, self.resize)),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(5),
                transforms.ToTensor(),
                # transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
            ])
        else:   # mode == 'test'
            self.transform = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.Resize((self.resize, self.resize)),
                transforms.ToTensor(),
                # transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
            ])
        self.path = os.path.join(root, 'ISIC2018')
#        print('csvdata!!!!!!!!!!!!!!!',os.path.join(root, mode+'.csv'))
        csvdata = self.loadCSV(os.path.join(root, mode+'.csv'))
#        print('csvdata',csvdata)
        self.data = []
        self.img2label = {}
        for i, (k,v) in enumerate(csvdata.items()):
#            print('K',k)
            self.data.append(v)  # list{ [file1.jpg, file2.jpg] , [file5.jpg, file9.jpg] ,... }
#            self.img2label[v] = i+startidx  # n45.454(9 digits):0
#        print('img2label',self.img2label)
        self.cls_num = len(self.data)
#        print('self.cls_num', self.cls_num )
        self.create_batch(self.batchsz)
    
    
    def loadCSV(self, csv_path):
        dicLabels = {}
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None) # skip (filename, label)
            for i, row in enumerate(csv_reader):
                filename = row[0]
                label = row[1]
                if label in dicLabels.keys():
                    dicLabels[label].append(filename)
                else:
                    dicLabels[label] = [filename]
        return dicLabels

    def create_batch(self, batchsz):
        self.support_x_batch = []
        self.query_x_batch = []
        self.mb_x_batch = []
        # for each task(batch)
        for b in range(batchsz):  # here batchsz=10000
            # 1. select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # replacement=False
            MB_class = (list(set(range(self.cls_num)).difference(selected_cls))) # the other 2 classes
            np.random.shuffle(MB_class)
            MB = []
            for cls in MB_class:
                selected_img_idx = np.random.choice(len(self.data[cls]),1,False) # only 1 image in MB each class
                index = np.array(selected_img_idx)
                MB.append(np.array(self.data[cls])[index].tolist())
            
            
            
            np.random.shuffle(selected_cls)
            support_x = [] 
            query_x = []
            for cls in selected_cls:
                # 2. select (k_shot + k_query) for each class
                selected_img_idx = np.random.choice(len(self.data[cls]), self.k_shot+self.k_query, False)
                np.random.shuffle(selected_img_idx)
                ### file_number -> filename ###
                indexDtrain = np.array(selected_img_idx[:self.k_shot])
                indexDtest = np.array(selected_img_idx[self.k_shot:])
                support_x.append( np.array(self.data[cls])[indexDtrain].tolist() )
                query_x.append( np.array(self.data[cls])[indexDtest].tolist() )
                ##############################
            # shuffle class number
            #random.shuffle(support_x)
            #random.shuffle(query_x)
            # add to batch list
            self.mb_x_batch.append(MB)
            self.support_x_batch.append(support_x)
            self.query_x_batch.append(query_x)

    def __getitem__(self, index):
        """
        Args:
        index: task_number
        """    
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        MB = torch.FloatTensor(2, 3, self.resize, self.resize)


        # get full filename
        flatten_support_x = [os.path.join(self.path, item) for sublist in \
            self.support_x_batch[index] for item in sublist]

        # get full filename
        flatten_query_x = [os.path.join(self.path, item) for sublist in \
            self.query_x_batch[index] for item in sublist]
        
        flatten_MB = [os.path.join(self.path, item) for sublist in \
            self.mb_x_batch[index] for item in sublist]

        support_y_relative = np.zeros(self.setsz)
        query_y_relative= np.zeros(self.querysz)
        
        for i in range(self.n_way):
            support_y_relative[self.k_shot*i:self.k_shot*(i+1)] = i
            query_y_relative[self.k_query*i:self.k_query*(i+1)] = i

        
        for i, path in enumerate(flatten_support_x):
            support_x[i] = self.transform(path)
        # print('supp',support_x.shape)
        for i, path in enumerate(flatten_query_x):
            query_x[i] = self.transform(path)
        # print('query',query_x.shape)    
        for i, path in enumerate(flatten_MB):
            # print('path',path)
            MB[i] = self.transform(path)
        # print('MB',MB.shape)    

        return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative),MB
    
    def __len__(self):
        return self.batchsz # return number of task
    

if __name__ == '__main__':
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    import time

#    plt.ion()
    mini = ISIC('F:\ICML2022\CityU\ISIC2018', mode='train', n_way=2, k_shot=1, k_query=15, batchsz=1000, resize=84)
    print('Done')
    for i, set_ in enumerate(mini):
        support_x, support_y, query_x, query_y, MB = set_
        print('support_x',support_x.shape)
        print('support_y',support_y)
        print('query_y',query_y)
        print('MB',MB.shape)
        
        support_x = make_grid(support_x, nrow=2)
        print('supp',support_x.transpose(2,0).numpy().shape)
        query_x = make_grid(query_x, nrow=2)

        plt.figure(1)
        plt.imshow(support_x.transpose(2,0).numpy())
#        plt.save('./1.jpg')
        plt.pause(0.5)

        plt.figure(2)
        plt.imshow(query_x.transpose(2,0).numpy())
        plt.pause(0.5)

        time.sleep(5)
        