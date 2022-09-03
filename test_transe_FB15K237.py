# -*- coding: UTF-8 -*-
'''
Created on 2021年8月12日 
@author: sunke
'''

import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

import platform

print(platform.architecture())

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")



# dataloader for training
train_dataloader = TrainDataLoader(
    in_path = "./benchmarks/FB15K237/", 
    nbatches = 30,
    threads = 8, 
    sampling_mode = "normal", 
    bern_flag = 1, 
    filter_flag = 1, 
    neg_ent = 25,
    neg_rel = 0)



# define the model
transe = TransE(
    ent_tot = train_dataloader.get_ent_tot(),
    rel_tot = train_dataloader.get_rel_tot(),
    dim = 20, 
    p_norm = 1, 
    norm_flag = True)


# test the model
transe.load_checkpoint('./checkpoint/transe.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = False)
tester.run_link_prediction(type_constrain = False)