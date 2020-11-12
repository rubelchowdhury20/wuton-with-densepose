# Standard library imports
import os

# Third party imports
from torch.nn import init

import shutil
import torch

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

# utility functions for training
def save_checkpoint(state, is_best, filename='checkpoint.pth'):
	"""Saves checkpoint to disk"""
	directory = "./weights/"
	if not os.path.exists(directory):
		os.makedirs(directory)
	filename = directory + filename
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, directory+'model_best.pth')

def ensure_folder(folder):
	if not os.path.exists(folder):
		os.makedirs(folder)

# initialize network weight
def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('Linear') != -1:
		init.normal(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		init.xavier_normal_(m.weight.data, gain=0.02)
	elif classname.find('Linear') != -1:
		init.xavier_normal_(m.weight.data, gain=0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
	# print('initialization method [%s]' % init_type)
	if init_type == 'normal':
		net.apply(weights_init_normal)
	elif init_type == 'xavier':
		net.apply(weights_init_xavier)
	elif init_type == 'kaiming':
		net.apply(weights_init_kaiming)
	else:
		raise NotImplementedError('initialization method [%s] is not implemented' % init_type)