import torch
import torch.autograd as autograd
import torch.nn as nn
from .Model import Model
import torchsnooper
import torch.nn.functional as F



class RotatE(Model):

	def __init__(self, ent_tot, rel_tot, dim = 100, margin = 6.0, epsilon = 2.0):
		super(RotatE, self).__init__(ent_tot, rel_tot)

		self.margin = margin
		self.epsilon = epsilon

		self.dim_e = dim * 2
		self.dim_r = dim

		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)

		self.ent_embedding_range = nn.Parameter(
			torch.Tensor([(self.margin + self.epsilon) / self.dim_e]), 
			requires_grad=False
		)


		nn.init.uniform_(
			tensor = self.ent_embeddings.weight.data, 
			a=-self.ent_embedding_range.item(), 
			b=self.ent_embedding_range.item()
		)

		self.rel_embedding_range = nn.Parameter(
			torch.Tensor([(self.margin + self.epsilon) / self.dim_r]), 
			requires_grad=False
		)

		nn.init.uniform_(
			tensor = self.rel_embeddings.weight.data, 
			a=-self.rel_embedding_range.item(), 
			b=self.rel_embedding_range.item()
		)

		self.margin = nn.Parameter(torch.Tensor([margin]))
		self.margin.requires_grad = False
		

	

		
# 	@torchsnooper.snoop()
	def _calc(self, h, t, r, mode):
		pi = self.pi_const
		#实部分和虚部, real number,  imaginary number 
		re_h, im_h = torch.chunk(h, 2, dim=-1)
		re_t, im_t = torch.chunk(t, 2, dim=-1)
		#Make phases of relations uniformly distributed in [-pi, pi]
		phase_relation = r / (self.rel_embedding_range.item() / pi)

		re_relation = torch.cos(phase_relation)    #real number
		im_relation = torch.sin(phase_relation)    #imaginary number

		
		#数据变换未做运算，原算法无这部分操作
		re_h = re_h.view(-1, re_relation.shape[0], re_h.shape[-1]).permute(1, 0, 2)  
		re_t = re_t.view(-1, re_relation.shape[0], re_t.shape[-1]).permute(1, 0, 2)
		im_h = im_h.view(-1, re_relation.shape[0], im_h.shape[-1]).permute(1, 0, 2)
		im_t = im_t.view(-1, re_relation.shape[0], im_t.shape[-1]).permute(1, 0, 2)
		im_relation = im_relation.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
		re_relation = re_relation.view(-1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)
		
		
		if mode == "head_batch":
			re_score = re_relation * re_t + im_relation * im_t
			im_score = re_relation * im_t - im_relation * re_t
			re_score = re_score - re_h
			im_score = im_score - im_h
		else:
			re_score = re_h * re_relation - im_h * im_relation   #弄清这里的矩阵运算   element-wise
			im_score = re_h * im_relation + im_h * re_relation
			re_score = re_score - re_t
			im_score = im_score - im_t
		# score 包含了 positive score 和 negative score.
		score = torch.stack([re_score, im_score], dim = 0)
		score = score.norm(dim = 0).sum(dim = -1)
		
		return score.permute(1, 0).flatten()
	
	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		
		score = self.margin - self._calc(h ,t, r, mode)
		return score

	def predict(self, data):
		score = -self.forward(data)
		return score.cpu().data.numpy()

	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2)) / 3
		return regul