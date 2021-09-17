
import torch.nn as nn
import torch.nn.functional as F
from network_blocks import ConvBlock, FCBlock

class end_network(nn.Module):
	def __init__(self, 
			img_shape = 256,
			features_shape = 1024,
			latent_shape = 1024,
			out_shape = 5,
			use_dropout = False):
		super().__init__()

		# first: the image input
		layers = []
		layers.append(ConvBlock(1, 16))
		layers.append(ConvBlock(16, 16))
		layers.append(ConvBlock(16, 32))
		layers.append(ConvBlock(32, 32))
		layers.append(ConvBlock(32, 64))
		layers.append(nn.Flatten())
		after_flatten = int(img_shape  * img_shape / 16)
		layers.append(nn.Linear(after_flatten, latent_shape))
		self.input_img = nn.Sequential(*layers)

		# second: the feature input
		shapes = [features_shape]
		shapes.append(int(0.5(features_shape + latent_shape)))
		shapes.append(latent_shape)
		self.input_features = FC_Block(shapes = shapes, use_dropout = use_dropout)

		# third: the compined part
		shapes = [2*latent_shape]
		shapes.append(latent_shape)
		shapes.append(out_shape)
		self.end = FC_Block(shapes = shapes, use_dropout = use_dropout)

	def forward(self, image, features):
		image = self.input_img(image)
		features = self.input_features(features)
		combined = torch.cat((image, features), dim = 1)
		out = self.end(combined)
		return out


class FullModel(nn.Module):
	def __init__(self, 
			unet, 
			feature_extractor, 
			end, 
			threshold = None,
			unet_trainable = False, 
			feature_extractor_trainable = False):
		super().__init__()	

		### Parts of the network
		# self.unet is a segmentation network that decides how important 
		# different parts of the network are. 
		# self.feature_extractor was trained on a larger dataset, but with the 
		# final fully connected layers removed to get deep features of the image

		self.unet = unet
		self.feature_extractor = feature_extractor
		self.end = end

		#### Set parts of the network to not trainable ####

		if not unet_trainable:
			for parameter in self.unet.parameters():
				parameter.requires_grad = False


		if not feature_extractor_trainable:
			for parameter in self.feature_extractor.parameters():
				parameter.requires_grad = False

		# threshold controls in which way the masks produced by unet are used
		self.threshold = threshold

	def forward(self, image):

		features = self.feature_extractor(image)

		mask = self.unet(image)
		if self.threshold is not None:
			mask = (mask > threshold).float()
		masked_image = mask * image

		out = self.end(masked_image, features)

		return out



