import os
import cv2
import numpy as np
from tqdm import tqdm

from typing import List

from COIGAN.training.data.datasets_loaders import JsonLineDatasetBase
from COIGAN.training.data.dataset_generators import JsonLineDatasetBaseGenerator
from COIGAN.training.data.dataset_inspectors import JsonLineDatasetInspector

def remove_empty_jsonl(dataset_folder, out_dataset_folder):
    """
    Script that aim to remove all the samples without objects from a dataset in JsonLine format.
    """

    # Process variables
    #dataset_folder = "/home/max/Desktop/Articolo_coigan/COIGAN-IROS-2024/datasets/severstal_steel_defect_dataset/test_IROS2024/tile_train_set"
    #out_dataset_folder = "/home/max/Desktop/Articolo_coigan/COIGAN-IROS-2024/datasets/severstal_steel_defect_dataset/test_IROS2024/tile_train_set_filtered"
    
    dataset_image_folder = os.path.join(dataset_folder, "data")
    out_image_folder = os.path.join(out_dataset_folder, "data")
    
    poligons_field = "polygons"

    # loading the input dataset
    input_dataset = JsonLineDatasetBase(
        metadata_file_path = os.path.join(dataset_folder, "dataset.jsonl"),
        index_file_path = os.path.join(dataset_folder, "index"),
        binary = True
    )
    input_dataset.on_worker_init()

    # create the output dataset
    output_dataset = JsonLineDatasetBaseGenerator(
        out_dataset_folder,
        dump_every = 10000,
        binary = True
    )
    os.makedirs(out_dataset_folder, exist_ok=True) # create the data folder
    os.makedirs(out_image_folder, exist_ok=True) # create the data folder (for the images)

    # iterate over the dataset and jump all the samples without objects
    pbar = tqdm(total = len(input_dataset), desc = "Taken: 0, Filtered: 0")
    filtered = 0
    taken = 0
    for sample in input_dataset:
        if len(sample[poligons_field]) > 0:
            output_dataset.insert(sample)

            #copy the corresponding image
            input_image_path = os.path.join(dataset_image_folder, sample["img"])
            output_image_path = os.path.join(out_image_folder, sample["img"])
            os.system(f"cp {input_image_path} {output_image_path}")

            taken += 1
        else:
            filtered += 1
        pbar.set_description(f"Taken: {taken}, Filtered: {filtered}")
        pbar.update(1)
    
    output_dataset.close()
    pbar.close()
    print("Done!")


def scale_all_images(
        target_folder,
        interpolation = cv2.INTER_LINEAR,
        out_tile_size = (256, 256)
    ):
    """
    Method that overwrite all the images in a folder with a scaled version of the original images.

    NOTE: about interpolation methods:
    - INTER_NEAREST: nearest-neighbor, best choice for masks scaling (enlarging or reducing).
    - INTER_LINEAR: bilinear interpolation, good for enlarging images.
    - INTER_AREA: resampling using pixel area relation, good for reducing images.

    Args:
        target_folder (str): the folder containing the images to scale.
        interpolation (int): the interpolation method to use.
        out_tile_size (tuple): the size of the output images.
    """
    print(f"Scaling images in {os.path.basename(target_folder)} with {interpolation} interpolation to {out_tile_size} size...")

    # map all the images in the folder
    images = [f for f in os.listdir(target_folder) if f.endswith(".jpg")]

    # iterate over the images and scale them
    for image in tqdm(images, desc = "Scaling images"):
        image_path = os.path.join(target_folder, image)
        img = cv2.imread(image_path)
        img = cv2.resize(img, out_tile_size, interpolation = interpolation)
        cv2.imwrite(image_path, img)


def enalrge_masks(
        target_folder,
        kernel_size = 2,
        iterations = 3
    ):
    """
    Method that enlarge all the masks in a folder.

    Args:
        target_folder (str): the folder containing the masks to enlarge.
        kernel_size (int): the size of the kernel to use for the dilation.
        iterations (int): the number of iterations to use for the dilation.
    """
    print(f"Dilating masks in {os.path.basename(target_folder)})...")

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # map all the images in the folder
    images = [f for f in os.listdir(target_folder) if f.endswith(".jpg")]

    # iterate over the images and scale them
    for image in tqdm(images, desc = "Dilating masks"):
        image_path = os.path.join(target_folder, image)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.dilate(img, kernel, iterations = iterations)
        cv2.imwrite(image_path, img)


def create_dataset_report(
    dataset_path: str,
    fields_to_inspect: List[str] = ["polygons"],
    binary: bool = True
):
    
    # check if file params.json is present, otherwise create it
    # the json is like "{"tile_size": [256, 1600]}" with the shapes of the images
    # in the data folder, consider all the images to have the same shape
    if not os.path.exists(os.path.join(dataset_path, "params.json")):
        print("Creating the params.json file...")
        images = [f for f in os.listdir(os.path.join(dataset_path, "data")) if f.endswith(".jpg")]
        img = cv2.imread(os.path.join(dataset_path, "data", images[0]))
        with open(os.path.join(dataset_path, "params.json"), "w") as f:
            f.write(f"{{\"tile_size\": {list(img.shape[:2])}}}")
        print("Done!")


    print("Creating the report of the dataset...")

    inspector = JsonLineDatasetInspector(
        dataset_path=dataset_path,
        fields_to_inspect=fields_to_inspect,
        binary=binary
    )

    inspector.inspect()
    inspector.dump_report()
    inspector.dump_raw_report()
    inspector.generate_graphs()

    print("Done!")


if __name__ == "__main__":

    """
    remove_empty_jsonl(
        dataset_folder = "/coigan/COIGAN-IROS-2024/datasets/severstal_steel_defect_dataset/test_IROS2024/tile_train_set",
        out_dataset_folder = "/coigan/COIGAN-IROS-2024/datasets/severstal_steel_defect_dataset/test_IROS2024/tile_train_set_filtered"
    )
    ""
    targets = [
        ("/coigan/COIGAN-IROS-2024/datasets/Conglomerate Concrete Crack Detection/Train/reduced_enlarged_orig_cccd/images/data", cv2.INTER_AREA, (256, 256)),
        #("/coigan/COIGAN-IROS-2024/datasets/Conglomerate Concrete Crack Detection/Train/reduced_defect_cccd/images", cv2.INTER_AREA, (256, 256)),
        #("/coigan/COIGAN-IROS-2024/datasets/Conglomerate Concrete Crack Detection/Test/reduced_enlarged_orig_cccd/masks", cv2.INTER_NEAREST, (256, 256)),
    ]
    for target in targets:
        scale_all_images(
            target_folder = target[0],
            interpolation = target[1],
            out_tile_size = target[2]
        )
    print("Done!")
    ""

    enalrge_masks(
        target_folder = "/coigan/COIGAN-IROS-2024/datasets/Conglomerate Concrete Crack Detection/Test/reduced_enlarged_orig_cccd/masks",
        kernel_size = 2,
        iterations = 5
    )
    """

    create_dataset_report("/coigan/COIGAN-IROS-2024/datasets/severstal_steel_defect_dataset/test_IROS2024/tile_train_set_filtered")

    ""