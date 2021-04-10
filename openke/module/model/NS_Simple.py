import torch
import torch.nn as nn
from .Model import Model


class NS_Simple(Model):
	
	def __init__(self, ent_tot, rel_tot, dim=100, pos_para=1, neg_para=1):
		super(NS_Simple, self).__init__(ent_tot, rel_tot)
		
		self.dim = dim
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
		self.rel_inv_embeddings = nn.Embedding(self.rel_tot, self.dim)
		
		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_inv_embeddings.weight.data)
		
		self.pos_para = pos_para
		self.neg_para = neg_para
	
	def _calc_avg(self, h, t, r, r_inv):
		return (torch.sum(h * r * t, -1) + torch.sum(h * r_inv * t, -1)) / 2
	
	def _calc_ingr(self, h, r, t):
		return torch.sum(h * r * t, -1)
	
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
		r_inv = self.rel_inv_embeddings(my_r)
		
		score = torch.sum(torch.mm(h.t(), h) * torch.mm(r.t(), r) * torch.mm(t.t(), t) + 2.0 * torch.mm(h.t(), h) *
						  torch.mm(r.t(), r_inv) * torch.mm(t.t(), t) + torch.mm(h.t(), h) * torch.mm(r_inv.t(), r_inv)
						  * torch.mm(t.t(), t)) * self.neg_para / 4.0
		bh = self.ent_embeddings(batch_h)
		bt = self.ent_embeddings(batch_t)
		br = self.rel_embeddings(batch_r)
		br_inv = self.rel_inv_embeddings(batch_r)
		loss = (torch.sum(bh * br * bt, -1) + torch.sum(bh * br_inv * bt, -1)) / 2
		score += (self.pos_para - self.neg_para) * torch.sum(loss ** 2) - self.pos_para * (
					torch.sum(loss) * 2.0) + self.pos_para * bh.shape[0]
		return score
	
	def regularization(self, data):
		batch_h = data['my_h']
		batch_t = data['my_t']
		batch_r = data['my_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		r_inv = self.rel_inv_embeddings(batch_r)
		regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2) + torch.mean(r_inv ** 2)) / 4
		return regul
	
	def predict(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		score = -self._calc_ingr(h, r, t)
		return score.cpu().data.numpy()
