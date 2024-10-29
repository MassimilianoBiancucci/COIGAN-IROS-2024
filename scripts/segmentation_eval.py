#!/usr/bin/env python3

import logging
import os

os.environ["HYDRA_FULL_ERROR"] = "1"

import torch

import hydra
from omegaconf import OmegaConf

from COIGAN.training.data.datasets_loaders import make_dataloader
from COIGAN.segmentation.segmentation_trainer import SegmentationTrainer

LOGGER = logging.getLogger(__name__)

# config selector
idx = 0
presets = [
    "test_bridge_ssdd_eval.yaml"
]

@hydra.main(config_path="../configs/segmentation_training/", config_name=presets[idx], version_base="1.1")
def main(config: OmegaConf):
    
    #resolve the config inplace
    OmegaConf.resolve(config)
    
    LOGGER.info(f'Config: {OmegaConf.to_yaml(config)}')
    
    OmegaConf.save(config, os.path.join(os.getcwd(), 'config.yaml')) # saving the configs to config.hydra.run.dir

    train(config)


def train(config):
    torch.cuda.set_device(config.device)

    # generate the dataset and wrap it in a dataloader
    #dataloader = make_dataloader(config)
    val_dataloader = make_dataloader(config, validation=True)

    trainer = SegmentationTrainer(
        config, 
        dataloader = None, #dataloader
        val_dataloader = val_dataloader
    )
    #trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()