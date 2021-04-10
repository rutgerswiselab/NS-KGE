import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from .Model import Model


class NS_ComplEx(Model):
	
	def __init__(self, ent_tot, rel_tot, dim=100, p_norm=1, norm_flag=True, margin=None, epsilon=None, pos_para=1, neg_para=1):
		super(NS_ComplEx, self).__init__(ent_tot, rel_tot)
		
		self.dim = dim
		self.ent_re_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.ent_im_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_re_embeddings = nn.Embedding(self.rel_tot, self.dim)
		self.rel_im_embeddings = nn.Embedding(self.rel_tot, self.dim)
		
		nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
		nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)
		
		self.pos_para = pos_para
		self.neg_para = neg_para
	
	def _calc(self, h_re, h_im, t_re, t_im, r_re, r_im):
		return torch.sum(
			h_re * t_re * r_re
			+ h_im * t_im * r_re
			+ h_re * t_im * r_im
			- h_im * t_re * r_im,
			-1
		)
	
	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		my_t = data['my_t']
		my_h = data['my_h']
		my_r = data['my_r']
		h_re = self.ent_re_embeddings(my_h)
		h_im = self.ent_im_embeddings(my_h)
		t_re = self.ent_re_embeddings(my_t)
		t_im = self.ent_im_embeddings(my_t)
		r_re = self.rel_re_embeddings(my_r)
		r_im = self.rel_im_embeddings(my_r)

		loss = (torch.sum(torch.mm(h_re.t(), h_re) * torch.mm(t_re.t(), t_re) * torch.mm(r_re.t(), r_re)
				+ torch.mm(h_im.t(), h_im) * torch.mm(t_im.t(), t_im) * torch.mm(r_re.t(), r_re)
				+ torch.mm(h_re.t(), h_re) * torch.mm(t_im.t(), t_im) * torch.mm(r_im.t(), r_im)
				+ torch.mm(h_im.t(), h_im) * torch.mm(t_re.t(), t_re) * torch.mm(r_im.t(), r_im)
				+ 2.0 * torch.mm(h_re.t(), h_im) * torch.mm(t_re.t(), t_im) * torch.mm(r_re.t(), r_re)
				+ 2.0 * torch.mm(h_re.t(), h_re) * torch.mm(t_re.t(), t_im) * torch.mm(r_re.t(), r_im)
				- 2.0 * torch.mm(h_re.t(), h_im) * torch.mm(t_re.t(), t_re) * torch.mm(r_re.t(), r_im)
				+ 2.0 * torch.mm(h_im.t(), h_re) * torch.mm(t_im.t(), t_im) * torch.mm(r_re.t(), r_im)
				- 2.0 * torch.mm(h_im.t(), h_im) * torch.mm(t_im.t(), t_re) * torch.mm(r_re.t(), r_im)
				- 2.0 * torch.mm(h_re.t(), h_im) * torch.mm(t_im.t(), t_re) * torch.mm(r_im.t(), r_im))) * self.neg_para
		
		if len(batch_h) > 0:
			h_re = self.ent_re_embeddings(batch_h)
			h_im = self.ent_im_embeddings(batch_h)
			t_re = self.ent_re_embeddings(batch_t)
			t_im = self.ent_im_embeddings(batch_t)
			r_re = self.rel_re_embeddings(batch_r)
			r_im = self.rel_im_embeddings(batch_r)
			score = torch.sum(h_re * t_re * r_re + h_im * t_im * r_re + h_re * t_im * r_im - h_im * t_re * r_im, -1)
			loss += self.pos_para * torch.sum((1.0 - score) ** 2) - self.neg_para * torch.sum(score ** 2)
		
		loss += self.regularization(data)
		return loss
	
	def regularization(self, data):
		batch_h = data['my_h']
		batch_t = data['my_t']
		batch_r = data['my_r']
		h_re = self.ent_re_embeddings(batch_h)
		h_im = self.ent_im_embeddings(batch_h)
		t_re = self.ent_re_embeddings(batch_t)
		t_im = self.ent_im_embeddings(batch_t)
		r_re = self.rel_re_embeddings(batch_r)
		r_im = self.rel_im_embeddings(batch_r)
		regul = (torch.mean(h_re ** 2) +
				 torch.mean(h_im ** 2) +
				 torch.mean(t_re ** 2) +
				 torch.mean(t_im ** 2) +
				 torch.mean(r_re ** 2) +
				 torch.mean(r_im ** 2)) / 6
		return regul
	
	def predict(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h_re = self.ent_re_embeddings(batch_h)
		h_im = self.ent_im_embeddings(batch_h)
		t_re = self.ent_re_embeddings(batch_t)
		t_im = self.ent_im_embeddings(batch_t)
		r_re = self.rel_re_embeddings(batch_r)
		r_im = self.rel_im_embeddings(batch_r)
		score = -torch.sum(h_re * t_re * r_re + h_im * t_im * r_re + h_re * t_im * r_im - h_im * t_re * r_im, -1)
		return score.cpu().data.numpy()
