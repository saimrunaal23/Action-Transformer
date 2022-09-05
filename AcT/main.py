# GENERAL LIBRARIES
import os
import argparse
from datetime import datetime
# MACHINE LEARNING LIBRARIES
import numpy as np
import tensorflow as tf
import torch as nn
# CUSTOM LIBRARIES
from utils.tools import read_yaml, Logger
from utils.trainer import Trainer


# LOAD CONFIG 
parser = argparse.ArgumentParser(description='Process some input')
parser.add_argument('--config', default='utils/config.yaml', type=str, help='Config path', required=False)    
parser.add_argument('--benchmark','-b', action='store_true', help='Run a benchmark') 
parser.add_argument('--search','-s', action='store_true', help='Run a random search')
    
args = parser.parse_args()
config = read_yaml(args.config)

for entry in ['MODEL_DIR','RESULTS_DIR','LOG_DIR']:
    if not os.path.exists(config[entry]):
        os.mkdir(config[entry])

now = datetime.now()
logger = Logger(config['LOG_DIR']+now.strftime("%y%m%d%H%M%S"))


# SET GPU 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = torch.cuda.current_device()
torch.cuda.set_device(gpus)


# SET TRAINER
trainer = Trainer(config, logger)

if args.benchmark:
    # RUN BENCHMARK
    trainer.do_benchmark()

elif args.search:
    # RUN RANDOM SEARCH
    trainer.do_random_search()
    
else:
    print('Nothing to do! Specify one of the following arguments:')
    print('\t --benchmark [-b]: run a benchmark')
    print('\t --search [-s]: run a random search')