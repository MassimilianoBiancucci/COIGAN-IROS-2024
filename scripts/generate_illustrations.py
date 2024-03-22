#!/usr/bin/env python3

import os
import logging
import json

os.environ["HYDRA_FULL_ERROR"] = "1"

import cv2
import torch

import hydra

from tqdm import tqdm
from omegaconf import OmegaConf

from COIGAN.utils.common_utils import sample_data
from COIGAN.training.data.datasets_loaders import make_dataloader
from COIGAN.inference.coigan_inference import COIGANinference
from COIGAN.evaluation.losses.fid.fid_score import calculate_fid_given_paths

LOGGER = logging.getLogger(__name__)


@hydra.main(config_path="../configs/evaluation/", config_name="test_gen.yaml", version_base="1.1")
def main(config: OmegaConf):

    #resolve the config inplace
    OmegaConf.resolve(config)

    # create the folder for the generated images
    out_path = config.generated_imgs_path
    os.makedirs(out_path, exist_ok=True)
    
    n_samples = config.n_samples
    dataloader = sample_data(make_dataloader(config))
    model = COIGANinference(config)
    idx = 0
    pbar = tqdm(total=n_samples)

    while True:
        # inference on the next sample
        sample = next(dataloader)
        inpainted_img = model(sample["gen_input"])

        # convert the input masks into a numpy array
        gen_input_union_mask = sample["gen_input_union_mask"].cpu().numpy()
        gen_input_union_mask = (gen_input_union_mask*255).astype("uint8")

        # convert the base image into a numpy array
        base_images = sample["base"].cpu().numpy().transpose(0, 2, 3, 1)
        base_images = (base_images*255).astype("uint8")

        # save the inpainted image in the target folder
        for base_img, gen_img, mask in zip(base_images, inpainted_img, gen_input_union_mask):
            cv2.imwrite(os.path.join(out_path, f"{idx}_base.png"), base_img)
            cv2.imwrite(os.path.join(out_path, f"{idx}_gen.png"), gen_img)
            cv2.imwrite(os.path.join(out_path, f"{idx}_mask.png"), mask)
            
            pbar.update()
            idx += 1

            # linux command to zip a folder in a file with a given name is: zip -r <zip_file_name> <folder_name>

            if idx >= n_samples:
                return


if __name__ == "__main__":
    main()