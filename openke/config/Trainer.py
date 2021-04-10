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

class Trainer(object):

	def __init__(self, 
				 model = None,
				 data_loader = None,
				 train_times = 1000,
				 alpha = 0.5,
				 use_gpu = True,
				 opt_method = "sgd",
				 save_steps = None,
				 checkpoint_dir = None,
				 on_step = True,
				 lr_decay = 0,
				 weight_decay = 0):

		self.work_threads = 8
		self.train_times = train_times

		self.opt_method = opt_method
		self.optimizer = None
		self.lr_decay = lr_decay
		self.weight_decay = weight_decay
		self.alpha = alpha

		self.model = model
		self.data_loader = data_loader
		self.use_gpu = use_gpu
		self.save_steps = save_steps
		self.checkpoint_dir = checkpoint_dir
		
		self.on_step = on_step

	def train_one_step(self, data):
		self.optimizer.zero_grad()
		if self.data_loader.sampling_mode == 'WholeSampling':
			loss = self.model({
				'batch_h': self.to_var(data['batch_h'], self.use_gpu),
				'batch_t': self.to_var(data['batch_t'], self.use_gpu),
				'batch_r': self.to_var(data['batch_r'], self.use_gpu),
				'my_t': self.to_var(np.array([i for i in range(self.data_loader.get_ent_tot())]), self.use_gpu),
				'my_h': self.to_var(np.array([i for i in range(self.data_loader.get_ent_tot())]), self.use_gpu),
				'my_r': self.to_var(np.array([i for i in range(self.data_loader.get_rel_tot())]), self.use_gpu),
				'mode': data['mode'],
			})
		else:
			loss = self.model({
				'batch_h': self.to_var(data['batch_h'], self.use_gpu),
				'batch_t': self.to_var(data['batch_t'], self.use_gpu),
				'batch_r': self.to_var(data['batch_r'], self.use_gpu),
				'batch_y': self.to_var(data['batch_y'], self.use_gpu),
				'mode': data['mode'],
			})
		if self.on_step:
			loss.backward()
			self.optimizer.step()
			return loss.item()
		else:
			return loss

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
			batch = 0
			for data in self.data_loader:
				loss = self.train_one_step(data)
				res += loss
				if self.data_loader.sampling_mode == 'mySampling' or self.data_loader.sampling_mode == 'myAnotherSampling':
					training_range.set_description("Epoch %d | Batch %d | loss: %f" % (epoch, batch, loss))
					batch += 1
			training_range.set_description("Epoch %d | loss: %f" % (epoch, res))
			# print("Epoch %d | loss: %f\n" % (epoch, res))
			if self.data_loader.sampling_mode == 'mySampling' or self.data_loader.sampling_mode == 'myAnotherSampling':
				print('\n')
			
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