import numpy as np

# DATASET PARAMETERS
DATASET = 'nyudv2'
TRAIN_DIR = './data/nyudv2'  # 'Modify data path'
VAL_DIR = TRAIN_DIR
TRAIN_LIST = './data/nyudv2/train.txt'
VAL_LIST = './data/nyudv2/val.txt'


SHORTER_SIDE = 350
CROP_SIZE = 500
RESIZE_SIZE = None

NORMALISE_PARAMS = [1./255,  # Image SCALE
                    np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),  # Image MEAN
                    np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)),  # Image STD
                    1./5000]  # Depth SCALE
BATCH_SIZE = 4
NUM_WORKERS = 16
NUM_CLASSES = 40
LOW_SCALE = 0.5
HIGH_SCALE = 2.0
IGNORE_LABEL = 255

# ENCODER PARAMETERS
ENC = '101'  # ResNet101
ENC_PRETRAINED = True  # pre-trained on ImageNet or randomly initialised

# GENERAL
FREEZE_BN = True
NUM_SEGM_EPOCHS = [100] * 3  # [150] * 3 if using ResNet152 as backbone
PRINT_EVERY = 10
RANDOM_SEED = 42
VAL_EVERY = 5  # how often to record validation scores

# OPTIMISERS' PARAMETERS
LR_ENC = [5e-4, 2.5e-4, 1e-4]  # TO FREEZE, PUT 0
LR_DEC = [3e-3, 1.5e-3, 7e-4]
MOM_ENC = 0.9  # TO FREEZE, PUT 0
MOM_DEC = 0.9
WD_ENC = 1e-5  # TO FREEZE, PUT 0
WD_DEC = 1e-5
LAMDA = 2e-4  # slightly better
BN_threshold = 2e-2  # slightly better
OPTIM_DEC = 'sgd'
