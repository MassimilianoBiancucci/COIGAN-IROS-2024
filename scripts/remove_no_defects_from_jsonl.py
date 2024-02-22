import os
from tqdm import tqdm

from COIGAN.training.data.datasets_loaders import JsonLineDatasetBase
from COIGAN.training.data.dataset_generators import JsonLineDatasetBaseGenerator

if __name__ == "__main__":
    """
    Script that aim to remove all the jsonl files that are empty
    """

    # Process variables
    dataset_folder = "/home/max/Desktop/Articolo_coigan/COIGAN-IROS-2024/datasets/severstal_steel_defect_dataset/test_IROS2024/tile_train_set"
    out_dataset_folder = "/home/max/Desktop/Articolo_coigan/COIGAN-IROS-2024/datasets/severstal_steel_defect_dataset/test_IROS2024/tile_train_set_filtered"
    
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