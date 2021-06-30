import sys
import os
from os.path import join
import numpy as np
import argparse
import torch
from src.transform import SSDTransformer
import cv2
from PIL import Image

from utils import generate_dboxes, Encoder
from models import *

from configs.utils import parse_config


def get_args():
    parser = argparse.ArgumentParser("Implementation of SSD")
    parser.add_argument("--single", action='store_true', help='Inference mode')
    parser.add_argument('--config', type=str, help='Model config')
    parser.add_argument("--pretrained-model", type=str, help='Trained checkpoint')
    parser.add_argument("--input-rgb", type=str, required=True, help="Path to input RGB image/dir")
    parser.add_argument("--input-lp", type=str, required=True, help="Path to input Local Pattern image/dir")
    parser.add_argument("--output", type=str, default=None, help="Path to output image/dir")
    args = parser.parse_args()
    return args


def test_single(cfg, pretrained_model, input_rgb_path, input_lp_path, output_path):
    model = SSD(backbone=ResNet())
    checkpoint = torch.load(pretrained_model)
    model.load_state_dict(checkpoint["model_state_dict"])
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    dboxes = generate_dboxes()
    transformer = SSDTransformer(dboxes, (300, 300), val=True)
    img = Image.open(input_rgb_path).convert("RGB")
    lp = Image.open(input_lp_path).convert("RGB")
    img, lp, _, _, _ = transformer(img, lp, None, torch.zeros(1,4), torch.zeros(1))
    encoder = Encoder(dboxes)

    if torch.cuda.is_available():
        img = img.cuda()
    with torch.no_grad():
        ploc, plabel = model([img.unsqueeze(dim=0), lp.unsqueeze(dim=0)])
        result = encoder.decode_batch(ploc, plabel, cfg.NMS_THRESHOLD, 20)[0]
        loc, label, prob = [r.cpu().numpy() for r in result]
        best = np.argwhere(prob > cfg.CLS_THRESHOLD).squeeze(axis=1)
        loc = loc[best]
        label = label[best]
        prob = prob[best]
        output_img = cv2.imread(input_rgb_path)
        if len(loc) > 0:
            height, width, _ = output_img.shape
            loc[:, 0::2] *= width
            loc[:, 1::2] *= height
            loc = loc.astype(np.int32)
            for box, lb, pr in zip(loc, label, prob):
                category = cfg.CLASSES[lb]
                color = cfg.COLORS[lb]
                xmin, ymin, xmax, ymax = box
                cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), color, 2)
                text_size = cv2.getTextSize(category + " : %.2f" % pr, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                cv2.rectangle(output_img, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color,
                              -1)
                cv2.putText(
                    output_img, category + " : %.2f" % pr,
                    (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1)
        if output_path is None:
            output = "{}_prediction.jpg".format(input_rgb_path[:-4])
        else:
            output = output_path
        cv2.imwrite(output, output_img)
        print(input_rgb_path + ' --> ' + output_path)


def test(cfg, args):
    if args.single:
        test_single(cfg, args.pretrained_model, args.input, args.output)
    else:
        image_ids = os.listdir(args.input)
        for img_id in image_ids:
            test_single(
                cfg, args.pretrained_model, 
                join(args.input_rgb, img_id), 
                join(args.input_lp, img_id), 
                join(args.output, img_id)
                )

            
if __name__ == "__main__":
    args = get_args()
    cfg = parse_config(args.config)
    test(cfg, args)