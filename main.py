# -*- coding: utf-8 -*-
import argparse
from baseline_zoo.utils.config import Config
from baseline_zoo.model.builder import build_model
from baseline_zoo.data.builder import build_data
from baseline_zoo.trainer.builder import build_trainer
from baseline_zoo.utils.args import process_args

args = argparse.ArgumentParser(description='the option of the BaselineZoo')
args.add_argument('--seed', type=int, default=0,
                  help='the random seed of the whole project')
args.add_argument('--gpu', type=str, default='0',
                  help='gpus to used')                    
args.add_argument('--config', type=str,
                  help='the config dir to use') 
args.add_argument('--accelerator', type=str, default='ddp', 
                  help='the type of accelerator')

args = args.parse_args()

if __name__ == '__main__':
    args = process_args(args)
    config = Config(args.config)
    model = build_model(config, args)
    data_pipeline = build_data(config, args)
    trainer = build_trainer(config, args)
    if config.train.lr == 'auto':
        trainer.tune(model, data_pipeline)
    trainer.fit(model, data_pipeline)