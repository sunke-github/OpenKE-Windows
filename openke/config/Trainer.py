# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm
import torchsnooper
from dgl.nn.pytorch import RelGraphConv
import dgl
import numpy as np
from datetime import datetime 


class Trainer(object):

	def __init__(self, 
				 model = None,
				 data_loader = None,
				 train_times = 1000,
				 alpha = 0.5,
				 use_gpu = True,
				 opt_method = "sgd",
				 save_steps = None,
				 checkpoint_dir = None):

		self.work_threads = 8
		self.train_times = train_times

		self.opt_method = opt_method
		self.optimizer = None
		self.lr_decay = 0
		self.weight_decay = 0
		self.alpha = alpha

		self.model = model
		self.data_loader = data_loader
		self.use_gpu = use_gpu
		self.save_steps = save_steps
		self.checkpoint_dir = checkpoint_dir

	def train_one_step(self, data):
		self.optimizer.zero_grad(set_to_none=True)   #.zero_grad(set_to_none=True)  2021-11-16  
		loss = self.model({
			'batch_h': self.to_var(data['batch_h'], self.use_gpu),
			'batch_t': self.to_var(data['batch_t'], self.use_gpu),
			'batch_r': self.to_var(data['batch_r'], self.use_gpu),
			'batch_y': self.to_var(data['batch_y'], self.use_gpu),
			'mode': data['mode']
		})
		loss.backward()
		self.optimizer.step()
		return loss.item()
# 	@torchsnooper.snoop()
	def train_one_step_mars(self, data):
		self.optimizer.zero_grad(set_to_none=True)   #.zero_grad(set_to_none=True)  2021-11-16  
# 		=====================================
		start=datetime.now()
		batch_h = self.to_var(data['batch_h'], self.use_gpu)
		batch_t = self.to_var(data['batch_t'], self.use_gpu)
		batch_r = self.to_var(data['batch_r'], self.use_gpu)
		mode = data['mode']
		src = batch_h
		dst = batch_t
		rel = batch_r		
		uniq_entity, edges = torch.unique(torch.cat((src, dst)), return_inverse=True)
		re_labled_edges = edges	
		if mode == "head_batch":
			src = edges[0:dst.shape[0]]
			dst = edges[-dst.shape[0]:]
		else:
			edges = edges[0:src.shape[0]*2]
			src, dst = torch.reshape(edges, (2, -1))		
		g, rel, norm = self.build_graph_from_triplets(len(uniq_entity), self.data_loader.get_rel_tot(),
											 (src, rel, dst))
		node_id = uniq_entity
		edge_type = rel
		edge_norm = self.node_norm_to_edge_norm(g,norm.view(-1, 1))
		end=datetime.now() 
# 		print((end-start).microseconds/1000)
		# 		=====================================
		loss = self.model({
			'batch_h': batch_h,
			'batch_t': batch_t,
			'batch_r': batch_r,
			'batch_y': self.to_var(data['batch_y'], self.use_gpu),
			'mode': data['mode'],
			'node_id': node_id,
			'edge_type':edge_type,
			'edge_norm':edge_norm,
			'g': g,
			're_labled_edges': re_labled_edges
			})
		loss.backward()
		self.optimizer.step()
		return loss.item()
	
	def run(self):
				
		if self.use_gpu:
			self.model.cuda()

		if self.optimizer != None:
			pass
		elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
			self.optimizer = optim.Adagrad(
				self.model.parameters(),
				lr=self.alpha,
				lr_decay=self.lr_decay,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
			self.optimizer = optim.Adadelta(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "Adam" or self.opt_method == "adam":
			self.optimizer = optim.Adam(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "AdamW" or self.opt_method == "adamw":   #2021-11-16
			self.optimizer = optim.AdamW(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		else:
			self.optimizer = optim.SGD(
				self.model.parameters(),
				lr = self.alpha,
				weight_decay=self.weight_decay,
			)
		print("Finish initializing...")
		
		
		training_range = tqdm(range(self.train_times))
		for epoch in training_range:
			res = 0.0
# 			print("size of data_load:",len(self.data_loader))
			
			for data in self.data_loader:    #在这里加载获取数据
# 				loss = self.train_one_step(data)
				loss = self.train_one_step(data)
				res += loss
# 				print(res)
			training_range.set_description("Epoch %d | loss: %f" % (epoch, res))
			
			if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
				print("Epoch %d has finished, saving..." % (epoch))
				self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))

	def set_model(self, model):
		self.model = model

	def to_var(self, x, use_gpu):
		if use_gpu:
			return Variable(torch.from_numpy(x).cuda())
		else:
			return Variable(torch.from_numpy(x))

	def set_use_gpu(self, use_gpu):
		self.use_gpu = use_gpu

	def set_alpha(self, alpha):
		self.alpha = alpha

	def set_lr_decay(self, lr_decay):
		self.lr_decay = lr_decay

	def set_weight_decay(self, weight_decay):
		self.weight_decay = weight_decay

	def set_opt_method(self, opt_method):
		self.opt_method = opt_method

	def set_train_times(self, train_times):
		self.train_times = train_times

	def set_save_steps(self, save_steps, checkpoint_dir = None):
		self.save_steps = save_steps
		if not self.checkpoint_dir:
			self.set_checkpoint_dir(checkpoint_dir)

	def set_checkpoint_dir(self, checkpoint_dir):
		self.checkpoint_dir = checkpoint_dir
		
		
		
	def get_scores(self):
		training_range = tqdm(range(self.train_times))
		
		totalScores = np.array([])
		for epoch in training_range:	
			for data in self.data_loader:    #在这里加载获取数据
				scores = self.model.get_scores({
				'batch_h': self.to_var(data['batch_h'], self.use_gpu),
				'batch_t': self.to_var(data['batch_t'], self.use_gpu),
				'batch_r': self.to_var(data['batch_r'], self.use_gpu),
				'batch_y': self.to_var(data['batch_y'], self.use_gpu),
				'mode': data['mode']
				})
				totalScores = np.append(totalScores,scores)
		return 	totalScores	
	
	
	def build_graph_from_triplets(self,num_nodes, num_rels, triplets):
		""" Create a DGL graph. The graph is bidirectional because RGCN authors
		use reversed relations.
		This function also generates edge type and normalization factor
		(reciprocal of node incoming degree)
		"""
		g = dgl.graph(([], []))
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		if self.use_gpu:
			g = g.to(device)
		g.add_nodes(num_nodes)
		src, rel, dst = triplets
		src, dst = torch.cat((src, dst)), torch.cat((dst, src))
		rel = torch.cat((rel, rel + num_rels))
		dst, indices = torch.sort(dst)
		src = src[indices]
		rel = rel[indices].to(torch.int64)
# 		=========================================
		g.add_edges(src, dst)
		norm = self.comp_deg_norm(g).to(torch.int64)
# 		print("# nodes: {}, # edges: {}".format(num_nodes, len(src)))
		return g, rel, norm
	
	def comp_deg_norm(self,g):
		g = g.local_var()
		in_deg = g.in_degrees(range(g.number_of_nodes())).float()
		norm = 1.0 / in_deg
		norm[torch.isinf(norm)] = 0
		return norm
	
	def node_norm_to_edge_norm(self,g, node_norm):
		g = g.local_var()
		# convert to edge norm
		g.ndata['norm'] = node_norm
		g.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
		return g.edata['norm']
				
				
				
			
			
			