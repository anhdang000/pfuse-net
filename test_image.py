"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import numpy as np
import argparse
import torch
from transform import SSDTransformer
import cv2
from PIL import Image
from utils import generate_dboxes, Encoder
from configs.utils import parse_config


def get_args():
    parser = argparse.ArgumentParser("Implementation of SSD")
    parser.add_argument("--image", type=str, required=True, help="the path to RGB image")
    parser.add_argument("--lp-image", type=str, required=True, help="the path to LP image")
    parser.add_argument("--cls-threshold", type=float, default=0.3)
    parser.add_argument('--config', type=str, help='Model config')
    parser.add_argument("--pretrained-model", type=str, default="trained_models/SSD.pth")
    parser.add_argument("--output", type=str, default=None, help="the path to output image")
    args = parser.parse_args()
    return args


def test(cfg, opt):
    model = cfg.MODEL(backbone=cfg.BACKBONE, cfg=cfg, num_classes=len(cfg.CLASSES))
    checkpoint = torch.load(opt.pretrained_model)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    dboxes = generate_dboxes()
    transformer = SSDTransformer(dboxes, (300, 300), val=True)
    img = Image.open(opt.image).convert("RGB")
    lp_img = Image.open(opt.lp_image).convert("RGB")
    img,lp_img, _, _, _ = transformer(img, lp_img,None, torch.zeros(1,4), torch.zeros(1))
    encoder = Encoder(dboxes)

    if torch.cuda.is_available():
        img = img.cuda()
        lp_img = lp_img.cuda()
    with torch.no_grad():
        ploc, plabel = model([img.unsqueeze(dim=0),lp_img.unsqueeze(dim=0)])
        result = encoder.decode_batch(ploc, plabel, opt.nms_threshold, 20)[0]
        loc, label, prob = [r.cpu().numpy() for r in result]
        best = np.argwhere(prob > opt.cls_threshold).squeeze(axis=1)
        loc = loc[best]
        label = label[best]
        prob = prob[best]
        output_img = cv2.imread(opt.image)
        if len(loc) > 0:
            height, width, _ = output_img.shape
            loc[:, 0::2] *= width
            loc[:, 1::2] *= height
            loc = loc.astype(np.int32)
            for box, lb, pr in zip(loc, label, prob):
                category = cfg.CLASSES[lb]
                color = cfg.CLASSES[lb]
                xmin, ymin, xmax, ymax = box
                cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), color, 2)
                text_size = cv2.getTextSize(category + " : %.2f" % pr, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                cv2.rectangle(output_img, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color,
                              -1)
                cv2.putText(
                    output_img, category + " : %.2f" % pr,
                    (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1)
        if opt.output is None:
            output = "{}_prediction.jpg".format(opt.image[:-4])
        else:
            output = opt.output
        cv2.imwrite(output, output_img)


if __name__ == "__main__":
    opt = get_args()
    cfg = parse_config(opt.config)
    test(cfg, opt)
