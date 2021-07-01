from models.resnet_parallel import *
from models.resnet_fusion import *


class SSDConcat(Base):
    def __init__(self, backbone=ResNetParallel(), cfg=None, num_classes=10, num_parallel=2):
        super().__init__()
        self.feature_extractor = backbone
        self.num_classes = num_classes
        self.num_parallel = num_parallel # TuyenNQ modified
        self._build_additional_features(self.feature_extractor.out_channels, self.num_parallel)
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.loc = []
        self.conf = []
        #TuyenNQ modified: Since step for predict bbox we only compute from rgb image so no need to Parallel here
        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            self.conf.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self.init_weights()

    def _build_additional_features(self, input_size, num_parallel):
        self.additional_blocks = []
        for i, (input_size, output_size, channels) in enumerate(
                zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):
            if i < 3:
                #TuyenNQ modified
                layer = nn.Sequential(
                    ModuleParallel(nn.Conv2d(input_size, channels, kernel_size=1, bias=False)),
                    BatchNorm2dParallel(channels, num_parallel),
                    ModuleParallel(nn.ReLU(inplace=True)),
                    #ModuleParallel(nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False)),
                    Concatenate(channels),
                    ModuleParallel(nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False)),
                    BatchNorm2dParallel(output_size, num_parallel),
                    ModuleParallel(nn.ReLU(inplace=True)),
                )
            else:
                layer = nn.Sequential(
                    ModuleParallel(nn.Conv2d(input_size, channels, kernel_size=1, bias=False)),
                    BatchNorm2dParallel(channels, num_parallel),
                    ModuleParallel(nn.ReLU(inplace=True)),
                    #ModuleParallel(nn.Conv2d(channels, output_size, kernel_size=3, bias=False)),
                    Concatenate(channels),
                    ModuleParallel(nn.Conv2d(channels, output_size, kernel_size=3, bias=False)),
                    BatchNorm2dParallel(output_size, num_parallel),
                    ModuleParallel(nn.ReLU(inplace=True)),
                )

            self.additional_blocks.append(layer)

        self.additional_blocks = nn.ModuleList(self.additional_blocks)


    def forward(self, x):
        x = self.feature_extractor(x)
        detection_feed = [x[0]] # Take only output of rgb to predict bbox
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x[0])
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)
        return locs, confs
