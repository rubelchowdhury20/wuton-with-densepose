# standard library imports
import os

# third party imports
import numpy as np
from PIL import Image

import torch.nn as nn
from torchvision import transforms


# local imports
import config
from . import utils
from . import geometric_transformer



class GeoTransformationInfer(nn.Module):
	def __init__(self, output_dir="./output/results"):
		super(GeoTransformationInfer, self).__init__()
		self.output_dir = output_dir
		utils.ensure_folder(self.output_dir)

	def forward(self, model_apparel, warped_image, model_image, warped_model_image, random_product_image, random_product_image_warped, output_on_random_product, batch_index, epoch):
		batch_size = warped_image.shape[0]

		model_apparel = model_apparel.cpu().numpy()
		warped_image = warped_image.cpu().numpy()
		model_image = model_image.cpu().numpy()
		warped_model_image = warped_model_image.cpu().numpy()
		random_product_image = random_product_image.cpu().numpy()
		random_product_image_warped = random_product_image_warped.cpu().numpy()
		output_on_random_product = output_on_random_product.cpu().numpy()

		for i in range(batch_size):
			self._save_image_sheet(
				batch_index*config.PARAMS["batch_size"] + i, 
				model_apparel[i], 
				warped_image[i], 
				model_image[i], 
				warped_model_image[i], 
				random_product_image[i],
				random_product_image_warped[i], 
				output_on_random_product[i], 
				epoch)

	def _save_image_sheet(self, 
		idx, 
		model_apparel, 
		warped_image, 
		model_image, 
		warped_model_image, 
		random_product_image,
		random_product_image_warped, 
		output_on_random_product, 
		epoch):

		# inverse normalization of the images along with channel first to channel last steps and finally converting np array to pillow format for saving
		model_apparel = np.moveaxis(model_apparel, 0, 2) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
		model_apparel = Image.fromarray(np.uint8(model_apparel * 255))
		warped_image = np.moveaxis(warped_image, 0, 2) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
		warped_image = Image.fromarray(np.uint8(warped_image * 255))
		model_image = np.moveaxis(model_image, 0, 2) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
		model_image = Image.fromarray(np.uint8(model_image * 255))
		warped_model_image = np.moveaxis(warped_model_image, 0, 2) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
		warped_model_image = Image.fromarray(np.uint8(warped_model_image * 255))
		random_product_image = np.moveaxis(random_product_image, 0, 2) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
		random_product_image = Image.fromarray(np.uint8(random_product_image * 255))
		random_product_image_warped = np.moveaxis(random_product_image_warped, 0, 2) * [0.229, 0.224, 0.225] + (0.485, 0.456, 0.406)
		random_product_image_warped = Image.fromarray(np.uint8(random_product_image_warped * 255))
		output_on_random_product = np.moveaxis(output_on_random_product, 0, 2) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
		output_on_random_product = Image.fromarray(np.uint8(output_on_random_product * 255))

		sheet = Image.new('RGB', (1568, 224), 'white')
		sheet.paste(model_apparel, (0, 0))
		sheet.paste(warped_image, (224, 0))
		sheet.paste(model_image, (448, 0))
		sheet.paste(warped_model_image, (672, 0))
		sheet.paste(random_product_image, (896, 0))
		sheet.paste(random_product_image_warped, (1120, 0))
		sheet.paste(output_on_random_product, (1344, 0))
		sheet.save(os.path.join(self.output_dir, "image_sheet_{}-epoch{}".format(idx, str(epoch).zfill(3)) + ".jpg"))

