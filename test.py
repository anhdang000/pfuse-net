from configs.utils import parse_config

if __name__ == '__main__':
	cfg = parse_config('configs/ssd_concat_kitti.py')
	print(cfg.MULTI_STEPS)