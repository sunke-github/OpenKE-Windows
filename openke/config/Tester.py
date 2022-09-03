# -*- coding: UTF-8 -*-

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
# from sklearn.metrics import roc_auc_score
import copy
from tqdm import tqdm
import torchsnooper
import platform
from datetime import datetime 


class Tester(object):

	def __init__(self, model = None, data_loader = None, use_gpu = True):		
		sys = platform.system().lower()
		if sys == "windows":
			base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.dll"))
		elif sys == "linux":
			base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
		
		
		self.lib = ctypes.cdll.LoadLibrary(base_file)
		self.lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
		self.lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
		self.lib.test_link_prediction.argtypes = [ctypes.c_int64]

		self.lib.getTestLinkMRR.argtypes = [ctypes.c_int64]
		self.lib.getTestLinkMR.argtypes = [ctypes.c_int64]
		self.lib.getTestLinkHit10.argtypes = [ctypes.c_int64]
		self.lib.getTestLinkHit3.argtypes = [ctypes.c_int64]
		self.lib.getTestLinkHit1.argtypes = [ctypes.c_int64]

		self.lib.getTestLinkMRR.restype = ctypes.c_float
		self.lib.getTestLinkMR.restype = ctypes.c_float
		self.lib.getTestLinkHit10.restype = ctypes.c_float
		self.lib.getTestLinkHit3.restype = ctypes.c_float
		self.lib.getTestLinkHit1.restype = ctypes.c_float

		self.model = model
		self.data_loader = data_loader
		self.use_gpu = use_gpu

		if self.use_gpu:
			self.model.cuda()

	def set_model(self, model):
		self.model = model

	def set_data_loader(self, data_loader):
		self.data_loader = data_loader

	def set_use_gpu(self, use_gpu):
		self.use_gpu = use_gpu
		if self.use_gpu and self.model != None:
			self.model.cuda()

	def to_var(self, x, use_gpu):
		if use_gpu:
			return Variable(torch.from_numpy(x).cuda())
		else:
			return Variable(torch.from_numpy(x))
	def test_one_step_mars(self, data):		  
		
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
		return self.model.predict({
			'batch_h': self.to_var(data['batch_h'], self.use_gpu),
			'batch_t': self.to_var(data['batch_t'], self.use_gpu),
			'batch_r': self.to_var(data['batch_r'], self.use_gpu),
			'mode': data['mode'],
			'node_id': node_id,
			'edge_type':edge_type,
			'edge_norm':edge_norm,
			'g': g,
			're_labled_edges': re_labled_edges
		})
	
	
	def test_one_step(self, data):		  
		return self.model.predict({
			'batch_h': self.to_var(data['batch_h'], self.use_gpu),
			'batch_t': self.to_var(data['batch_t'], self.use_gpu),
			'batch_r': self.to_var(data['batch_r'], self.use_gpu),
			'mode': data['mode']
		})
		
		
		
		
#	  @torchsnooper.snoop()
	def run_link_prediction(self, type_constrain = False):
		self.lib.initTest()
		self.data_loader.set_sampling_mode('link')
		if type_constrain:
			type_constrain = 1
		else:
			type_constrain = 0
		training_range = tqdm(self.data_loader)
		for index, [data_head, data_tail] in enumerate(training_range):
			
			score = self.test_one_step(data_head)   
			self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)	  
			
			score = self.test_one_step(data_tail)
			self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)
			
		self.lib.test_link_prediction(type_constrain)

		mrr = self.lib.getTestLinkMRR(type_constrain)
		mr = self.lib.getTestLinkMR(type_constrain)
		hit10 = self.lib.getTestLinkHit10(type_constrain)
		hit3 = self.lib.getTestLinkHit3(type_constrain)
		hit1 = self.lib.getTestLinkHit1(type_constrain)
		print (hit10)
		return mrr, mr, hit10, hit3, hit1

	def get_best_threshlod(self, score, ans):
		res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
		order = np.argsort(score)
		res = res[order]

		total_all = (float)(len(score))
		total_current = 0.0
		total_true = np.sum(ans)
		total_false = total_all - total_true

		res_mx = 0.0
		threshlod = None
		for index, [ans, score] in enumerate(res):    #可以从这里提取正样本的 scores
			if ans == 1:
				total_current += 1.0
			res_current = (2 * total_current + total_false - index - 1) / total_all
			if res_current > res_mx:
				res_mx = res_current
				threshlod = score
		return threshlod, res_mx

	def run_triple_classification(self, threshlod = None):
		self.lib.initTest()
		self.data_loader.set_sampling_mode('classification')
		score = []
		ans = []
		training_range = tqdm(self.data_loader)
		for index, [pos_ins, neg_ins] in enumerate(training_range):
			res_pos = self.test_one_step(pos_ins)
			ans = ans + [1 for i in range(len(res_pos))]
			score.append(res_pos)

			res_neg = self.test_one_step(neg_ins)
			ans = ans + [0 for i in range(len(res_pos))]
			score.append(res_neg)

		score = np.concatenate(score, axis = -1)
		ans = np.array(ans)

		if threshlod == None:
			threshlod, _ = self.get_best_threshlod(score, ans)

		res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
		order = np.argsort(score)
		res = res[order]

		total_all = (float)(len(score))
		total_current = 0.0
		total_true = np.sum(ans)
		total_false = total_all - total_true
#===============================================
# 		for index, [ans, score] in enumerate(res):
			
# 			if score > threshlod:
# 				acc = (2 * total_current + total_false - index) / total_all
# 				break
# 			elif ans == 1:
# 				total_current += 1.0
# 		print(acc,threshlod)
# 		return acc, threshlod	
#=============================================================	
		tp = 0.0
		fn = 0.0
		fp = 0.0
		tn = 0.0
# 		结果会阈值的影响
		for index, [ans, score] in enumerate(res):
			if score > threshlod and ans==1:
				fn += 1.0    # tp
			elif score > threshlod and ans==0:
				tn += 1.0    # fp
			elif score <= threshlod and ans==1:
				tp += 1.0    # fn
			elif score <= threshlod and ans==0:
				fp += 1.0    # tn	
# 			#==================================================================
		print(tp,fn,fp,tn)
  
		acc = (tp +tn)/(tp +tn+fn+fp)
		precision = tp/(tp+fp)
		recall = tp/(tp+fn)
		f1_score = 2*precision*recall/(precision+recall)
		
		print(acc,precision,recall,f1_score)
		return acc,precision,recall,f1_score
# ========================================		
	def get_scores(self):
		self.lib.initTest()
		self.data_loader.set_sampling_mode('classification')
		score = []
		ans = []
		training_range = tqdm(self.data_loader)
		
		totalPScores = []
		totalNScores = []
		
		for index, [pos_ins, neg_ins] in enumerate(training_range):
			res_pos = self.test_one_step(pos_ins)
			ans = ans + [1 for i in range(len(res_pos))]
			totalPScores.append(res_pos)

			res_neg = self.test_one_step(neg_ins)
			ans = ans + [0 for i in range(len(res_pos))]
			totalNScores.append(res_neg)

		totalPScores = np.array(totalPScores).reshape(-1,1)
		totalNScores = np.array(totalNScores).reshape(-1,1)
		return totalPScores,totalNScores 
	
	
	
	
	
	
	
	
	
	
	
	
	