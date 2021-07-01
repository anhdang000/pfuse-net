from models.resnet_parallel import *
from models.resnet_fusion import *
class SSDExchange(Base):
    def __init__(self, backbone=ResNetParallel(), cfg=None, num_classes=10, num_parallel=2, bn_threshold=1e-2):
        super().__init__()
        self.feature_extractor = backbone
        self.num_classes = num_classes
        self.num_parallel = num_parallel # TuyenNQ modified
        self.bn_threshold = bn_threshold
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
                conv1 = ModuleParallel(nn.Conv2d(input_size, channels, kernel_size=1, bias=False))
                bn1 = BatchNorm2dParallel(channels, num_parallel)
                relu = ModuleParallel(nn.ReLU(inplace=True))
                conv2 = ModuleParallel(nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False))
                bn2 = BatchNorm2dParallel(output_size, num_parallel)
                bn2_list = []
                for module in bn2.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        bn2_list.append(module)
                layer = nn.Sequential(
                    conv1,
                    bn1,
                    relu,
                    #ModuleParallel(nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False)),
                    conv2,
                    bn2,
                    Exchange(bn2_list, bn_threshold=self.bn_threshold),
                    relu,
                )
            else:
                conv1 = ModuleParallel(nn.Conv2d(input_size, channels, kernel_size=1, bias=False))
                bn1 = BatchNorm2dParallel(channels, num_parallel)
                relu = ModuleParallel(nn.ReLU(inplace=True))
                conv2 = ModuleParallel(nn.Conv2d(channels, output_size, kernel_size=3, bias=False))
                bn2 = BatchNorm2dParallel(output_size, num_parallel)
                bn2_list = []
                for module in bn2.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        bn2_list.append(module)
                layer = nn.Sequential(
                    conv1,
                    bn1,
                    relu,
                    #ModuleParallel(nn.Conv2d(channels, output_size, kernel_size=3, bias=False)),
                    conv2,
                    bn2,
                    Exchange(bn2_list, bn_threshold=self.bn_threshold),
                    relu,
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
