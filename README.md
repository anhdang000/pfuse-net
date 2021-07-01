## Train from scratch
```Shell
python train.py --connfig configs/ssd_concat_kitti.py
```

## Resume training
```Shell
python train.py --connfig configs/ssd_concat_kitti.py --resume PATH_TO_CHECKPOINT
```

## Inference
### Single
```Shell
python inference.py --single --config CONFIG --pretrained-model CHECKPOINT --input-rgb IMAGE_FILE --input-lp LP_IMAGE_FILE --output OUTPUT_FILE
```

### Multiple
```Shell
python inference.py --config CONFIG --pretrained-model CHECKPOINT --input-rgb IMAGE_DIR --input-lp LP_IMAGE_DIR --output OUTPUT_DIR
```
