# Third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn import init

# local imports
import config
DEVICE = config.DEVICE

# INIT_TYPE = "kaiming"

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
	elif init_type == 'zeros':
		net.apply(weights_init_zeros)
	else:
		raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def featureL2Norm(feature):
	epsilon = 1e-6
	norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
	return torch.div(feature,norm)

# loading the resnet-18 model and removing the fully connected layers to preserve the spatial dimension details
class FeatureExtraction(nn.Module):
	def __init__(self, normalization=True, use_cuda=True):
		super(FeatureExtraction, self).__init__()
		self.normalization = normalization
		self.use_cuda = use_cuda
		self.model = models.resnet18(pretrained=False).to(DEVICE)
		self.model = nn.Sequential(*list(self.model.children())[:-2])
		init_weights(self.model, init_type="normal")
		# move to GPU
		if self.use_cuda:
			self.model = self.model.to(DEVICE)
		
	def forward(self, image_batch):
		features = self.model(image_batch)
		if self.normalization:
			features = featureL2Norm(features)
		return features

	def freeze_layers(self):
		for param in self.model.parameters():
			param.requires_grad = False

	def unfreeze_layers(self):
		for param in self.model.parameters():
			param.requires_grad = True
		ct = 0
		for name, child in self.model.named_children():
			ct += 1
			if ct < 6:
				for name2, parameters in child.named_parameters():
					parameters.requires_grad = False

# model to lower the number of channels to 3 for feeding the agnostic model with key points to the resnet-18 model
class ReduceNumberOfChannels(nn.Module):
	def __init__(self, use_cuda=True):
		super(ReduceNumberOfChannels, self).__init__()
		self.use_cuda = use_cuda
		self.model = nn.Sequential(
            # input is 24 x 224 x 224
            nn.Conv2d(6, 3, 1, 1, 0, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(3))
		if self.use_cuda:
			self.model = self.model.to(DEVICE)

	def forward(self, image_batch):
		return self.model(image_batch)


# the correlation module 
class FeatureCorrelation(nn.Module):
	def __init__(self,normalization=True,matching_type='correlation'):
		super(FeatureCorrelation, self).__init__()
		self.normalization = normalization
		self.matching_type=matching_type
		self.ReLU = nn.ReLU()
	
	def forward(self, feature_A, feature_B):
		b,c,h,w = feature_A.size()
		if self.matching_type=='correlation':
			# reshape features for matrix multiplication
			feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
			feature_B = feature_B.view(b,c,h*w).transpose(1,2)
			# perform matrix mult.
			feature_mul = torch.bmm(feature_B,feature_A)
			# indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]
			correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
			if self.normalization:
				correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))
			return correlation_tensor

		if self.matching_type=='subtraction':
			return feature_A.sub(feature_B)
		
		if self.matching_type=='concatenation':
			return torch.cat((feature_A,feature_B),1)

# the regression module to get the theta outputs for transformation
class FeatureRegression(nn.Module):
	def __init__(self,output_dim=6, use_cuda=True):
		super(FeatureRegression, self).__init__()
		self.output_dim = output_dim

		# self.conv1 = nn.Conv2d(196, 64, kernel_size=3)
		# self.bn1 = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(49, 32, kernel_size=3)
		self.bn2 = nn.BatchNorm2d(32)
		# self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=0)
		# self.bn3 = nn.BatchNorm2d(32)
		# self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=0)
		# self.bn4 = nn.BatchNorm2d(16)
		# self.linear1 = nn.Linear(3200, 512)
		# self.linear2 = nn.Linear(512, 6)

		# Regressor for the 3 * 2 affine matrix
		self.fc_loc = nn.Sequential(
			nn.Linear(800, 256),
			nn.ReLU(True),
			nn.Linear(256, 128),
			nn.ReLU(True),
			nn.Linear(128, self.output_dim)
		)

		# Initialize the weights/bias with identity transformation
		if(self.output_dim == 6):
			self.fc_loc[4].weight.data.zero_()
			self.fc_loc[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
		# elif(self.output_dim == 18):
		# 	self.fc_loc[4].weight.data.zero_()
		# 	self.fc_loc[4].bias.data.copy_(torch.tensor([0]*18, dtype=torch.float))



	def forward(self, x):
		# x = self.conv1(x)
		# x = self.bn1(x)
		# x = nn.ReLU()(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = nn.ReLU()(x)
		# x = self.conv3(x)
		# x = self.bn3(x)
		# x = nn.ReLU()(x)
		# x = self.conv4(x)
		# x = self.bn4(x)
		# x = nn.ReLU()(x)
		x = x.view(x.size(0), -1)
		# x = self.linear1(x)
		# x = nn.ReLU()(x)
		# x = self.linear2(x)
		x = self.fc_loc(x)
		x = F.relu(x)

		return x


class GenerateTheta(nn.Module):
	def __init__(self,geometric_model="affine", tps_grid_size=3, use_cuda=True):
		super(GenerateTheta, self).__init__()
		self.geometric_model = geometric_model
		self.tps_grid_size = tps_grid_size
		self.use_cuda = use_cuda
		self.FeatureExtraction = FeatureExtraction()
		self.ReduceNumberOfChannels = ReduceNumberOfChannels()
		self.FeatureCorrelation = FeatureCorrelation()
		if(self.geometric_model == "affine"):
			self.FeatureRegression = FeatureRegression(output_dim=6)
		elif(self.geometric_model == "tps"):
			self.FeatureRegression = FeatureRegression(output_dim=2*(self.tps_grid_size ** 2))
		if use_cuda:
			self.ReduceNumberOfChannels = self.ReduceNumberOfChannels.to(DEVICE)
			self.FeatureExtraction = self.FeatureExtraction.to(DEVICE)
			self.FeatureCorrelation = self.FeatureCorrelation.to(DEVICE)
			self.FeatureRegression = self.FeatureRegression.to(DEVICE)
		self.ReLU = nn.ReLU(inplace=True)

	def freeze_layers(self):
		self.FeatureExtraction.freeze_layers()

	def unfreeze_layers(self):
		self.FeatureExtraction.unfreeze_layers()

	def forward(self, product_image_batch, model_image_batch):
		# feature extraction
		feature_A = self.FeatureExtraction(product_image_batch)
		model_image_batch = self.ReduceNumberOfChannels(model_image_batch)
		feature_B = self.FeatureExtraction(model_image_batch)
		feature_A = featureL2Norm(feature_A)
		feature_B = featureL2Norm(feature_B)

		# feature correlation
		correlation = self.FeatureCorrelation(feature_A, feature_B)

		# regression for transformation parameters
		theta = self.FeatureRegression(correlation)

		return theta