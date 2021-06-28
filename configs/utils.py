import sys
import os

def parse_config(config):
    config_name = config.split('/')[-1].split('.')[0]
    sys.path.append(config.split('/')[0])

    cfg = __import__(config_name, globals(), locals(), [], 0)

    return cfg