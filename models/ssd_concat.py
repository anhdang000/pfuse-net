from models.resnet_parallel import *
from models.resnet_fusion import *
from collections import OrderedDict
from itertools import islice
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

    def init_load_from(self, checkpoint):
        with torch.no_grad():
            weights = self.feature_extractor.feature_extractor.state_dict()
            fe_checkpoints = []
            for (k1, v1), (k2, v2) in zip(checkpoint["model_state_dict"].items(), weights.items()):
                # print(k)
                if "feature_extractor" in k1:
                    fe_checkpoints.append((k2, v1))
            # for k,v in feature_extractor.state_dict().items():
            #   weights =
            fe_checkpoints = OrderedDict(fe_checkpoints)
            self.feature_extractor.feature_extractor.load_state_dict(fe_checkpoints, strict=False)
            add_weights = []
            for i in range(0, 5):
                weights = islice(checkpoint["model_state_dict"].items(), 258, 318)
                name = "additional_blocks." + str(i)
                # print(name)
                count = 0
                for (k, v) in weights:
                    # print(k)
                    if name in k:
                        if count == 6:
                            add_weights.append(("added." + str(i), torch.tensor(0)))
                        add_weights.append((k, v))
                        count += 1

            add_weights = OrderedDict(add_weights)
            weights = []
            for (k1, v1), (k2, v2) in zip(add_weights.items(), self.additional_blocks.state_dict().items()):
                if "added" in k1:
                    weights.append((k2, v2))
                else:
                    weights.append((k2, v1))

            add_block_weights = OrderedDict(weights)
            self.additional_blocks.load_state_dict(add_block_weights)
            loc_weights = (list(checkpoint["model_state_dict"].items())[318:330])
            loc_w = []
            for (k1, v1), (k2, v2) in zip(loc_weights, self.loc.state_dict().items()):
                loc_w.append((k2, v1))
            loc_w = OrderedDict(loc_w)
            self.loc.load_state_dict(loc_w)
            # Since the difference of num classes, we cannot load conf module
            # conf_weights = (list(checkpoint["model_state_dict"].items())[330:342])
            # conf_w = []
            # for (k1,v1), (k2,v2) in zip(conf_weights, self.conf.state_dict().items()):
            #   conf_w.append((k2, v1))
            # conf_w = OrderedDict(conf_w)
            # self.conf.load_state_dict(conf_w)
    def forward(self, x):
        x = self.feature_extractor(x)
        detection_feed = [x[0]] # Take only output of rgb to predict bbox
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x[0])
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)
        return locs, confs
