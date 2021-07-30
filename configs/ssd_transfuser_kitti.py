from models import *


NAME = 'ssd_transfuser_kitti'


# ---------------- MODEL -------------------------------
BACKBONE = ResNetParallel_NonSharing()
MODEL = SSDTransfuser


# ---------------- KITTI DATASET -----------------------
DATASET = 'KITTI'

CLASSES = [
    '_background_', 'Car', 'Van', 'Truck', 'Pedestrian',
    'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare'
    ]

COLORS = [
    None, (39, 129, 113), (21, 35, 42), (49, 119, 155), (7, 185, 124),
    (46, 34, 146), (105, 184, 169), (22, 18, 5), (147, 71, 73), (181, 64, 91)
    ]


# ---------------- DATASET PATH ------------------------
ROOT = '../stereo_datasets'
IMAGE_DIR = 'image_2'
LABEL_DIR = 'label_2'
LPIMAGE_DIR = 'lp_image'


#______________PRETRAINED________________________
PRETRAINED_DIR = "./coco/pretrained/SSD.pth"
USE_COCO_PRETRAIN = False



# ---------------- LOGGING -----------------------------
SAVE_FOLDER = 'trained_models'
SAVE_INTERVAL = 10
LOG_PATH = 'tensorboard/SSD'


# ---------------- TRAINING CONFIGURATIONS -------------
NUM_EPOCHS = 60
BATCH_SIZE = 16
MULTI_STEPS = [40, 52]
LR = 2.6e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NMS_THRESHOLD = 0.5
NUM_WORKERS = 4

LOCAL_RANK = 0
#________________GPT PARAMETER_________________
VERT_ANCHORS = 8
HORZ_ANCHORS = 8
ANCHORS = VERT_ANCHORS * HORZ_ANCHORS
BLOCK_EXP = 4
N_LAYER = 8
N_HEAD = 4
N_SCALE = 4
EMBD_PDROP = 0.1
RESID_PDROP = 0.1
ATTN_PDROP = 0.1

