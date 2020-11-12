# standard library imports
import os
import math
import random
import argparse

# third party imports
import pickle
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils import data
import torch.nn as nn
from torch.nn import init
from torchsummary import summary

# local imports
import config
from modules import data_loader
from modules import utils
from modules import inference
from modules import geometric_transformer
from modules import losses

from modules.models import geometric_model
from modules.models import unet_model
from modules.models import gan_model


# global variables
DEVICE = config.DEVICE
BEST_LOSS = 0
EPOCH = 0

def main(args):
	global DEVICE
	global BEST_LOSS
	global EPOCH

	# path to all the required files
	image_source_dir = "./dataset/images/"
	seg_info_path = "./dataset/files/seg_info.pkl"
	densepose_info_path = "./dataset/files/densepose_info.pkl"
	image_pairs_path = "./dataset/files/product_model_list_final.pkl"

	# initializing the required data variables 
	with open(seg_info_path, "rb") as f:
		seg_info = pickle.load(f)

	with open(densepose_info_path, "rb") as f:
		densepose_info = pickle.load(f)

	with open(image_pairs_path, "rb") as f:
		image_pairs = pickle.load(f)

	# splitting the image-pairs to train and validation set
	# random.shuffle(image_pairs)
	train_image_pairs = image_pairs[:math.floor(len(image_pairs) * 0.95)]
	val_image_pairs = image_pairs[math.ceil(len(image_pairs) * 0.95):]

	# initializing the data loader for generating images for training
	train_dataset = data_loader.WutonDataset(train_image_pairs, image_source_dir, seg_info, densepose_info)
	train_loader = data.DataLoader(train_dataset, **config.PARAMS)

	# val_dataset = data_loader.WutonDataset(val_image_pairs, image_source_dir, seg_info, densepose_info, split="val")
	# val_loader = data.DataLoader(val_dataset, **config.PARAMS)

	infer_dataset = data_loader.WutonDataset(val_image_pairs[:20], image_source_dir, seg_info, densepose_info, split="val")
	infer_loader = data.DataLoader(infer_dataset, **config.PARAMS_VAL)

	# Establish convention for real and fake labels during training
	real_label = 1
	fake_label = 0

	thetaGeneratorAffine = geometric_model.GenerateTheta(geometric_model="affine")
	thetaGeneratorTPS = geometric_model.GenerateTheta(geometric_model="tps", tps_grid_size=config.TPS_GRID_SIZE)

	geo_tnf_affine = geometric_transformer.GeometricTransformer(geometric_model="affine")
	geo_tnf_tps = geometric_transformer.GeometricTransformer(geometric_model="tps", tps_grid_size=config.TPS_GRID_SIZE)

	unet = unet_model.UNet(3)
	unet = unet.to(DEVICE)

	discriminator_net = gan_model.Discriminator()
	discriminator_net = discriminator_net.to(DEVICE)

	#initialising all the networks with intial weights
	thetaGeneratorAffine.apply(utils.weights_init_normal)
	thetaGeneratorTPS.apply(utils.weights_init_normal)

	discriminator_net.apply(utils.weights_init_normal)

	# optimizer_affine = optim.SGD(thetaGeneratorAffine.parameters(), lr=config.LR, momentum=0.9)
	# optimizer_tps = optim.SGD(thetaGeneratorTPS.parameters(), lr=config.LR, momentum=0.9)
	# optimizer_unet = optim.SGD(unet.parameters(), lr=config.LR, momentum=0.9)
	# optimizer_discriminator = optim.SGD(discriminator_net.parameters(), lr=config.LR, momentum=0.9)

	optimizer_affine = optim.Adam(thetaGeneratorAffine.parameters(), lr=config.LR, betas=(0.5, 0.999))
	optimizer_tps = optim.Adam(thetaGeneratorTPS.parameters(), lr=config.TPS_LR, betas=(0.5, 0.999))
	optimizer_unet = optim.Adam(unet.parameters(), lr=config.LR, betas=(0.5, 0.999))
	optimizer_discriminator = optim.Adam(discriminator_net.parameters(), lr=config.LR, betas=(0.5, 0.999))

	criterion = nn.L1Loss()
	perceptual_criterion = losses.VGGLoss()
	gan_criterion = nn.BCELoss()

	geo_tnf_losses = utils.AverageMeter()
	unet_losses = utils.AverageMeter()
	perceptual_losses = utils.AverageMeter()

	disc_losses = utils.AverageMeter()
	gen_losses = utils.AverageMeter()

	D_x_count = utils.AverageMeter()
	D_G_z1_count = utils.AverageMeter()
	D_G_z2_count = utils.AverageMeter()

	train_losses = utils.AverageMeter()
	val_losses = utils.AverageMeter()

	# initiating the GeoTransformationInfer class for inference
	geo_transform_infer = inference.GeoTransformationInfer()

	# resuming the training if the weights are saved
	filename = "./weights/checkpoint.pth"
	if config.RESUME:
		if os.path.isfile(filename):
			print("=> loading checkpoint '{}'".format(filename))
			checkpoint = torch.load(filename)
			affine_state_dict = checkpoint["state_dict"]["affine"]
			tps_state_dict = checkpoint["state_dict"]["tps"]
			unet_state_dict = checkpoint["state_dict"]["unet"]
			discriminator_state_dict = checkpoint["state_dict"]["discriminator"]
			BEST_LOSS = checkpoint["best_loss"]
			EPOCH = checkpoint['epoch']

			thetaGeneratorAffine.load_state_dict(affine_state_dict)
			thetaGeneratorTPS.load_state_dict(tps_state_dict)
			unet.load_state_dict(unet_state_dict)
			discriminator_net.load_state_dict(discriminator_state_dict)
		else:
			print("=> no checkpoint found at '{}'".format(filename))

	for epoch in range(config.EPOCHS)[EPOCH+1:]:
		thetaGeneratorAffine.train()
		thetaGeneratorTPS.train()
		unet.train()
		discriminator_net.train()

		# training steps
		for batch_idx, (product_image_batch, \
					gan_product_image_batch, \
					model_image_batch, \
					model_seg_image_batch, \
					model_bodyshape_batch, \
					model_apparel_batch, \
					model_dense_u_batch, \
					model_dense_v_batch) in tqdm(enumerate(train_loader)):

			product_image_batch = product_image_batch.to(DEVICE)
			gan_product_image_batch = gan_product_image_batch.to(DEVICE)
			model_image_batch = model_image_batch.to(DEVICE)
			model_seg_image_batch = model_seg_image_batch.to(DEVICE)
			model_bodyshape_batch = model_bodyshape_batch.to(DEVICE)
			model_apparel_batch = model_apparel_batch.to(DEVICE)
			model_dense_u_batch = model_dense_u_batch.to(DEVICE)
			model_dense_v_batch = model_dense_v_batch.to(DEVICE)

			# concatenation of model agnostic image with densepose and background
			model_agnostic_image_batch = torch.cat([model_bodyshape_batch, model_seg_image_batch, model_dense_u_batch, model_dense_v_batch], dim=1)

			# zero the parameter gradients
			optimizer_affine.zero_grad()
			optimizer_tps.zero_grad()
			optimizer_unet.zero_grad()

			b_size = product_image_batch.shape[0]


			######################################################################################################
			# The first step is to trian the discriminator on real model images along with product
			######################################################################################################
			label = torch.full((b_size,), real_label, device=DEVICE)
			# Forward pass real model images through discriminator
			discriminator_output_real = discriminator_net(model_image_batch).view(-1)
			# calculate loss on all real images
			error_discriminator_real = gan_criterion(discriminator_output_real, label)
			# calculate gradients for Discriminator in backward pass
			error_discriminator_real.backward()
			D_x = discriminator_output_real.mean().item()
			D_x_count.update(D_x)


			######################################################################################################
			# second step is to train the discriminator on fake unet generated images
			######################################################################################################

			# generate fake images using unet and tps transformation
			# as in this case we don't want the weights to get updated apart from the discriminator
			# so converting the netwroks to eval mode along with torch.no_grad
			thetaGeneratorAffine.eval()				# disabling dropout, batch-norm and similar operations
			thetaGeneratorTPS.eval()	
			unet.eval()

			with torch.no_grad():					# stopping the network from updating parameters, only inference
				# affine transformation step
				theta_affine = thetaGeneratorAffine(gan_product_image_batch, model_agnostic_image_batch)
				affine_output = geo_tnf_affine(gan_product_image_batch, theta_affine)
				# tps transformation step
				theta_tps = thetaGeneratorTPS(affine_output, model_agnostic_image_batch)
				tps_output = geo_tnf_tps(affine_output, theta_tps)
				# unet transformation step
				unet_output = unet(gan_product_image_batch, model_agnostic_image_batch, theta_tps)
			
			label = torch.full((b_size,), fake_label, device=DEVICE)
			# Forward pass fake model images through discriminator
			discriminator_output_fake = discriminator_net(unet_output).view(-1)
			# calculate loss on all fake unet generated images
			error_discriminator_fake = gan_criterion(discriminator_output_fake, label)
			# calculate gradients for discriminator in backward pass for this batch
			error_discriminator_fake.backward()
			D_G_z1 = discriminator_output_fake.mean().item()
			D_G_z1_count.update(D_G_z1)


			# add the error for both real and fake images
			error_discriminator = error_discriminator_real + error_discriminator_fake
			disc_losses.update(error_discriminator, b_size)

			optimizer_discriminator.step()

			######################################################################################################
			# third step is the generator step
			######################################################################################################
			thetaGeneratorAffine.train()			# enabling dropout, batch_norm and other similar operations
			thetaGeneratorTPS.train()
			unet.train()

			label.fill_(real_label)  # fake labels are real for generator cost

			# affine transformation step
			theta_affine = thetaGeneratorAffine(product_image_batch, model_agnostic_image_batch)
			affine_output = geo_tnf_affine(product_image_batch, theta_affine)

			# tps transformation step
			theta_tps = thetaGeneratorTPS(affine_output, model_agnostic_image_batch)
			tps_output = geo_tnf_tps(affine_output, theta_tps)

			# unet transformation step
			unet_output = unet(product_image_batch, model_agnostic_image_batch, theta_tps)

			discriminator_output_fake = discriminator_net(unet_output).view(-1)
			# calculate generator loss based on this output
			error_generator = gan_criterion(discriminator_output_fake, label)
			D_G_z2 = discriminator_output_fake.mean().item()
			D_G_z2_count.update(D_G_z2)

			geo_tnf_loss = criterion(tps_output, model_apparel_batch)
			geo_tnf_losses.update(geo_tnf_loss, b_size)

			unet_loss = criterion(unet_output, model_image_batch)
			unet_losses.update(unet_loss, b_size)

			perceptual_loss = perceptual_criterion(unet_output, model_image_batch)
			perceptual_losses.update(perceptual_loss, b_size)

			generator_loss = gan_criterion(discriminator_output_fake, label)
			gen_losses.update(generator_loss, b_size)

			# batch_loss = geo_tnf_loss + unet_loss + 0.2*perceptual_loss + generator_loss
			batch_loss = geo_tnf_loss + 0.8*unet_loss + 0.3*perceptual_loss + generator_loss

			train_losses.update(batch_loss, b_size)

			batch_loss.backward()

			optimizer_affine.step()
			optimizer_tps.step()
			optimizer_unet.step()

			
			if batch_idx % 10 == 0:
				print("Train Progress--\t"
					"Train Epoch: {} [{}/{}]\t"
					"Total_loss:{:.4f} ({:.4f})\t"
					"Geo_tnf_loss:{:.4f} ({:.4f})\t"
					"Unet_loss:{:.4f} ({:.4f})\t"
					"perceptual_loss:{:.4f} ({:.4f})".format(epoch, batch_idx * b_size, len(train_loader.dataset),
					 											train_losses.val, train_losses.avg,
					 											geo_tnf_losses.val, geo_tnf_losses.avg,
					 											unet_losses.val, unet_losses.avg,
					 											perceptual_losses.val, perceptual_losses.avg))
				print("Adversarial losses--\t"
					"Generator Loss: {:.4f} ({:.4f})\t"
					"Discriminator Loss: {:.4f} ({:.4f})\t"
					"D_X: {:.4f} ({:.4f})\t"
					"D_G_z1: {:.4f} ({:.4f})\t"
					"D_G_z2: {:.4f} ({:.4f})".format(gen_losses.val, gen_losses.avg,
														disc_losses.val, disc_losses.avg,
														D_x_count.val, D_x_count.avg,
														D_G_z1_count.val, D_G_z1_count.avg,
														D_G_z2_count.val, D_G_z2_count.avg))
				print("\n")

		with torch.no_grad():
			print("Inferring the validation examples for output visualization.")
			for batch_idx, (product_image_batch, \
					gan_product_image_batch, \
					model_image_batch, \
					model_seg_image_batch, \
					model_bodyshape_batch, \
					model_apparel_batch, \
					model_dense_u_batch, \
					model_dense_v_batch) in tqdm(enumerate(infer_loader)):

				product_image_batch = product_image_batch.to(DEVICE)
				gan_product_image_batch = gan_product_image_batch.to(DEVICE)
				model_image_batch = model_image_batch.to(DEVICE)
				model_seg_image_batch = model_seg_image_batch.to(DEVICE)
				model_bodyshape_batch = model_bodyshape_batch.to(DEVICE)
				model_apparel_batch = model_apparel_batch.to(DEVICE)
				model_dense_u_batch = model_dense_u_batch.to(DEVICE)
				model_dense_v_batch = model_dense_v_batch.to(DEVICE)

				# concatenation of model agnostic image with densepose and background
				model_agnostic_image_batch = torch.cat([model_bodyshape_batch, model_seg_image_batch, model_dense_u_batch, model_dense_v_batch], dim=1)

				# getting output for ground truth product image
				theta_affine_gt = thetaGeneratorAffine(product_image_batch, model_agnostic_image_batch)
				affine_output_gt = geo_tnf_affine(product_image_batch, theta_affine_gt)

				theta_tps_gt = thetaGeneratorTPS(affine_output_gt, model_agnostic_image_batch)
				tps_output_gt = geo_tnf_tps(affine_output_gt, theta_tps_gt)

				unet_output_gt = unet(product_image_batch, model_agnostic_image_batch, theta_tps_gt)

				# getting output for randomly choosen product image
				theta_affine_new = thetaGeneratorAffine(gan_product_image_batch, model_agnostic_image_batch)
				affine_output_new = geo_tnf_affine(gan_product_image_batch, theta_affine_new)

				theta_tps_new = thetaGeneratorTPS(affine_output_new, model_agnostic_image_batch)
				tps_output_new = geo_tnf_tps(affine_output_new, theta_tps_new)

				unet_output_new = unet(gan_product_image_batch, model_agnostic_image_batch, theta_tps_new)


				geo_transform_infer(model_apparel_batch, tps_output_gt, model_image_batch, unet_output_gt, gan_product_image_batch, tps_output_new, unet_output_new, batch_idx, epoch)

		# remember best loss and save checkpoint
		is_best = val_losses.avg < BEST_LOSS
		utils.save_checkpoint({
			'state_dict': {"affine":thetaGeneratorAffine.state_dict(),
				"tps":thetaGeneratorTPS.state_dict(), 
				"unet":unet.state_dict(),
				"discriminator":discriminator_net.state_dict()},
			'best_loss': BEST_LOSS,
			'epoch': epoch
		}, is_best)







if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	parser.add_argument(
		"--mode",
		type=str,
		default="train")

	main(parser.parse_args())