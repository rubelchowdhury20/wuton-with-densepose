# standard library imports
import os
import sys
import random

# third party imports
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

sys.path.append("/media/tensor/EXTDRIVE/projects/virtual-try-on/densepose/detectron2/projects/DensePose")
from densepose.data.structures import DensePoseResult

# local imports
import config

DEVICE = config.DEVICE

class WutonDataset(Dataset):
	def __init__(self, image_pairs, source_dir, seg_info, densepose_info, split="train"):
		self.image_pairs = image_pairs
		self.source_dir = source_dir
		self.seg_info = seg_info
		self.densepose_info = densepose_info
		self.transformer = config.data_transforms[split]
		self.transform_mask_wNorm = transforms.Compose([transforms.ToPILImage(),
														transforms.Resize((224, 224)),
														transforms.ToTensor(),
														transforms.Normalize((0.5,), (0.5,))])

	def get_iuv_arr(self, pose_data):
		result_encoded = pose_data.results[0]
		iuv_arr = DensePoseResult.decode_png_data(*result_encoded)
		return iuv_arr


	def __getitem__(self, index):
		product_image_name = self.image_pairs[index][0]
		gan_product_image_name = random.choice(self.image_pairs)[0]
		model_image_name = self.image_pairs[index][1]

		product_image = Image.open(os.path.join(self.source_dir, product_image_name))
		gan_product_image = Image.open(os.path.join(self.source_dir, gan_product_image_name))
		model_image =  Image.open(os.path.join(self.source_dir, model_image_name))

		model_dense_info = self.densepose_info[model_image_name]
		model_dense_bbox = model_dense_info["bbox"][0]
		model_dense = self.get_iuv_arr(model_dense_info["densepose"])


		image_width, image_height = model_image.size

		product_image = np.array(product_image)[...,:3]
		gan_product_image = np.array(gan_product_image)[...,:3]
		model_image = np.array(model_image)[...,:3]

		product_image = self.transformer(product_image)
		gan_product_image = self.transformer(gan_product_image)
		model_image = self.transformer(model_image)
		model_dense_u = self.transform_mask_wNorm(model_dense[1])
		model_dense_v = self.transform_mask_wNorm(model_dense[2])

		product_seg_info = self.seg_info[product_image_name]
		gan_product_seg_info = self.seg_info[gan_product_image_name]
		model_seg_info = self.seg_info[model_image_name]

		product_seg_mask = torch.tensor(product_seg_info == 5).unsqueeze(0).repeat(3, 1, 1)
		gan_product_seg_mask = torch.tensor(gan_product_seg_info == 5).unsqueeze(0).repeat(3, 1, 1)

		model_background_mask = ~torch.tensor(model_seg_info == 0).unsqueeze(0).repeat(3, 1, 1)
		model_background_mask_single_channel = ~torch.tensor(model_seg_info == 0)
		model_seg_mask = (~torch.tensor(model_seg_info == 5) &
							~torch.tensor(model_seg_info == 14) &
							~torch.tensor(model_seg_info == 15)).unsqueeze(0).repeat(3, 1, 1)
		model_apparel_seg_mask = torch.tensor(model_seg_info == 5).unsqueeze(0).repeat(3, 1, 1)

		product_masked = product_image * product_seg_mask
		gan_product_masked = gan_product_image * gan_product_seg_mask
		model_image = model_image * model_background_mask

		model_bodyshape = model_background_mask_single_channel * 255
		model_bodyshape = self.transform_mask_wNorm(np.uint8(model_bodyshape))
		# model_bodyshape = self.transformer(np.uint8(model_bodyshape))
		model_bodyshape = F.interpolate(model_bodyshape, size=int(224/16))
		model_bodyshape = F.interpolate(model_bodyshape, size=224)

		model_seg_masked = model_image * model_background_mask * model_seg_mask
		model_apparel_masked = model_image * model_apparel_seg_mask

		return product_masked, gan_product_masked, model_image, model_seg_masked, model_bodyshape, model_apparel_masked, model_dense_u, model_dense_v		

	def __len__(self):
		return len(self.image_pairs)