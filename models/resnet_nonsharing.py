from models.base import *
from models.modules import *
from torchvision.models.resnet import resnet50


class ResNetParallel_NonSharing(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(pretrained=True)
        self.out_channels = [1024, 512, 512, 256, 256, 256]
        # TuyenNQ modified
        layers_backbone = [ModuleParallel_NonSharing(child, num_parallel=2) for child in list(backbone.children())[:7]]
        self.feature_extractor = nn.Sequential(*layers_backbone)

        conv4_block1 = self.feature_extractor[-1].module[0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x