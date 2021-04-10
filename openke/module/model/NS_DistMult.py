import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from .Model import Model


class NS_DistMult(Model):
	
	def __init__(self, ent_tot, rel_tot, dim=100, p_norm=1, norm_flag=True, margin=None, epsilon=None, pos_para=1, neg_para=1):
		super(NS_DistMult, self).__init__(ent_tot, rel_tot)
		
		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon
		self.norm_flag = norm_flag
		self.p_norm = p_norm
		
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
		
		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		else:
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
			)
			nn.init.uniform_(
				tensor=self.ent_embeddings.weight.data,
				a=-self.embedding_range.item(),
				b=self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor=self.rel_embeddings.weight.data,
				a=-self.embedding_range.item(),
				b=self.embedding_range.item()
			)
		
		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False
		
		self.pos_para = pos_para
		self.neg_para = neg_para
	
	def _calc(self, h, t, r, mode):
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
		if mode == 'head_batch':
			score = h * (r * t)
		else:
			score = (h * r) * t
		score = torch.sum(score, -1).flatten()
		return score
	
	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		my_t = data['my_t']
		my_h = data['my_h']
		my_r = data['my_r']
		h = self.ent_embeddings(my_h)
		t = self.ent_embeddings(my_t)
		r = self.rel_embeddings(my_r)
		
		loss = torch.sum(torch.mm(h.t(), h) * torch.mm(r.t(), r) * torch.mm(t.t(), t)) * self.neg_para
		
		if len(batch_h) > 0:
			bh = self.ent_embeddings(batch_h)
			bt = self.ent_embeddings(batch_t)
			br = self.rel_embeddings(batch_r)
			score = torch.sum(bh * br * bt, -1)
			loss += (self.pos_para - self.neg_para) * torch.sum(score ** 2) - self.pos_para * (torch.sum(score) * 2.0) + self.pos_para * bh.shape[0]
		
		return loss
	
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
	
	def predict(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		if mode == 'head_batch':
			score = h * (r * t)
		else:
			score = (h * r) * t
		score = - torch.sum(score, -1)
		return score.cpu().data.numpy()
