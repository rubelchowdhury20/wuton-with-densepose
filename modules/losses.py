# third party imports
import torch
import torch.nn as nn

# local imports
import config
from .models import perceptual_model

class VGGLoss(nn.Module):
	def __init__(self, layids = None):
		super(VGGLoss, self).__init__()
		self.vgg = perceptual_model.Vgg19()
		self.vgg.to(config.DEVICE)
		self.vgg.eval()

		self.criterion = nn.L1Loss()
		self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
		self.layids = layids

	def forward(self, x, y):
		x_vgg, y_vgg = self.vgg(x), self.vgg(y)
		loss = 0
		if self.layids is None:
			self.layids = list(range(len(x_vgg)))
		for i in self.layids:
			loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
		return loss