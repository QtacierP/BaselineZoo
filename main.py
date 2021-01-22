# -*- coding: utf-8 -*-
import argparse
import importlib
import sys
import os
import baseline_zoo as bz
from baseline_zoo.utils.config import Config
from baseline_zoo.model.builder import build_model
from baseline_zoo.data.builder import build_data

args = argparse.ArgumentParser(description='the option of the BaselineZoo')
args.add_argument('--seed', type=int, default=0,
                          help='the random seed of the whole project')
args.add_argument('--gpu', type=str, default='0',
                          help='gpus to used')                    
args.add_argument('--config', type=str,
                          help='the config dir to use') 

args = args.parse_args()

if __name__ == '__main__':
    config = Config(args.config)
    model = build_model(config)
    data_pipeline = build_data(config)
    model.fit(data_pipeline)