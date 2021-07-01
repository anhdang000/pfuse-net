from models.base import *
from models.modules import ModuleParallel, SqueezeAndExciteFusionAdd, Concatenate, Exchange
from torchvision.models.resnet import resnet50


class ResNetFuse(nn.Module):
	def __init__(self, multiple_fuse=None, method=None):
		super().__init__()
		backbone = resnet50(pretrained=True)
		self.out_channels = [1024, 512, 512, 256, 256, 256]
		self.multiple_fuse = multiple_fuse
		self.method = method
		self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])
		conv4_block1 = self.feature_extractor[-1][0]
		conv4_block1.conv1.stride = (1, 1)
		conv4_block1.conv2.stride = (1, 1)
		conv4_block1.downsample[0].stride = (1, 1)


	def forward(self, x):
		'''
		Input tensors `x`: [rgb, lp]
		'''
		for layer in self.feature_extractor:
			x = ModuleParallel(layer)(x)
			if self.multiple_fuse:
				if self.method == "concat":
					x = Concatenate(x[0].shape[1])(x)
				if self.method == "align_ESA":
					x = SqueezeAndExciteFusionAdd(x[0].shape[1])(x)

		return x