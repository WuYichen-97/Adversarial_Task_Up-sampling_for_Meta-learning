# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 20:36:12 2022

@author: coltonwu
"""

import torch
state_dict = torch.load('./pcn27000')
torch.save(state_dict,'pcn27000',_use_new_zipfile_serialization=False)