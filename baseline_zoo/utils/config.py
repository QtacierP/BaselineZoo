# -*- coding: utf-8 -*-
from bunch import Bunch
import importlib.util
import os 

class Config():
    # Warning: Please maintain the temp_dir to avoid import error
    def __init__(self, config_dir, temp_dir='temp'):
       self.config_dir = config_dir
       self.temp_dir = temp_dir
       self.init_config_list = ['data_configs', 'model_configs', 'train_configs']
       self._load_dict()
       self._bunch_dict()
    
    def _load_dict(self):
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        os.system('cp {} {}'.format(self.config_dir, self.temp_dir))
        config_file_name = os.path.split(self.config_dir)[-1].split('.')[0]
        config_module = importlib.import_module(self.temp_dir + '.' + config_file_name)
        for config in self.init_config_list:
            setattr(self, config, getattr(config_module, config))
        os.system('rm -rf {}'.format(self.temp_dir))

    def _bunch_dict(self):
        for config in self.init_config_list:
            sub_name = config.split('_')[0]
            setattr(self, sub_name, Bunch(getattr(self, config)))
    

    

