import os
import cv2
import torch
import numpy as np

from torchvision.io import read_image

from typing import Tuple, Union, Dict, List

from COIGAN.training.data.datasets_loaders.jsonl_dataset import JsonLineDatasetMasksOnly
from COIGAN.training.data.augmentation.augmentor import Augmentor


class JsonLineDatasetSegm(JsonLineDatasetMasksOnly):
    """
    Extension of the JsonLineDatasetMasksOnly class,
    This object aim to load the dataset samples in a format suitable for a segmentation task.
    So extend the JsonLineDatasetMasksOnly class to return the masks and the image of the sample.
    """

    def __init__(
        self,
        image_folder_path: str,
        metadata_file_path: str,
        index_file_path: str,
        classes: List[str],
        augmentor: Augmentor = None,
        masks_fields: List[str] = ["polygons"],
        mask_value: int = 1,
        size: Union[Tuple[int, int], int] = (256, 256),
        points_normalized: bool = False,
        binary: bool = False
    ):
        """
        the expected Json structure for each sub json line is: (See JsonLineDatasetMasksOnly)

        Args:
            image_folder_path (str): path to the folder containing the images
            metadata_path (str): metadata file containing the masks of each image
            index_path (str): index file containing the start position of each sample in the metadata file
            masks_fields (list[str]): list of the fields containing the masks, used to group the masks in the output dict, and to read the polygons.
            classes (Union[list[str],None], optional): list of the classes to load. Defaults to None. if None, all the classes will be loaded.
            size (Union[Tuple[int, int], int], optional): size of the output masks. Defaults to (256, 256).
            points_normalized (bool, optional): if the points are normalized. Defaults to False. if the are in the range [0, 1], the points_normalized parameter must be set to True.
            binary (bool, optional): if the metadata file is binary. Defaults to False.

        Raises:
            RuntimeError: _description_
            RuntimeError: _description_
        """

        super(JsonLineDatasetSegm, self).__init__(
            metadata_file_path, 
            index_file_path,
            masks_fields,
            classes,
            size,
            points_normalized,
            binary
        )

        self.image_folder_path = image_folder_path
        self.augmentor = augmentor
        self.mask_value = mask_value

        # check if the image folder exists
        if not os.path.isdir(self.image_folder_path):
            raise RuntimeError(f"Image folder path {self.image_folder_path} does not exist")
    

    def _reformat_masks(self, masks: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """
        Method that reformat the masks to a single tensor with the classes as channels
        with a final shape given by [n_classes, h, w]

        Args:
            masks (Dict[str, Dict[str, np.ndarray]]): the masks of the sample, 
                as dict with as keys the passed list of fields and as values other dicts with as keys 
                the classes and as values the masks as numpy arrays.

        Returns:
            np.ndarray: the reformatted masks with shape [n_classes, h, w]
        """

        # create the final tensor with the classes as channels
        final_masks = np.zeros((len(self.classes), self.size[0], self.size[1]), dtype=np.uint8)

        # for each field
        for _class in self.classes:
            for field in self.masks_fields:
                if _class in masks[field]:
                    final_masks[self.classes.index(_class), :, :] = masks[field][_class]
        
        return final_masks


    def _get_image_and_masks(self, idx: int, ret_meta: bool = False) -> Tuple[np.ndarray, dict]:
        """
        Method that return the image, the masks of the sample at the given index
        and eventually the metadata of the sample, if ret_meta is True.

        Args:
            idx (int): the index of the sample
            ret_meta (bool, optional): Flag that active the return of the metadata. Defaults to False.

        Returns:
            Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]: the image and the masks of the sample, 
                as dict with as keys the passed list of fields and as values other dicts with as keys 
                the classes and as values the masks as numpy arrays.

            Tuple[np.ndarray, dict, dict]: the image, the masks of the sample in the same format as 
                above and the raw metadata of the sample.
        """

        masks, metadata = self._get_masks(idx, ret_meta=True, mask_value=self.mask_value)

        # load the image
        #img = cv2.imread(os.path.join(self.image_folder_path, metadata["img"]))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # better way to load image directly in tensor format
        img = read_image(os.path.join(self.image_folder_path, metadata["img"]))

        if img is None:
            raise RuntimeError(f"Image {metadata['img']} not found")

        # reformat the masks to a single tensor with the classes as channels
        # with a final shape given by [h, w, n_classes]
        masks = self._reformat_masks(masks) # /255.0 not needed, if self.mask_value is 1
        masks = torch.as_tensor(masks, dtype=torch.float32)
        
        # apply the augmentations
        if self.augmentor is not None:
            img, masks = self.augmentor(img, masks)

        # apply normalization to img
        img = img.float() / 255.0

        if ret_meta:
            return {"inp": img, "out": masks, "meta": metadata}
        else:
            return {"inp": img, "out": masks}


    def __getitem__(self, idx: int) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Method that return the image and the masks of the sample at the given index.

        Args:
            idx (int): the index of the sample

        Returns:
            Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]: the image and the masks of the sample, 
                as dict with as keys the passed list of fields and as values other dicts with as keys 
                the classes and as values the masks as numpy arrays.
        """
        if isinstance(idx, slice):
            return [self._get_image_and_masks(i) for i in range(*idx.indices(len(self)))]
        else:
            return self._get_image_and_masks(idx)


    def __iter__(self):
        """
        Return an iterator over the dataset
        """
        for i in range(len(self)):
            yield self._get_image_and_masks(i)



if __name__ == "__main__":

    from COIGAN.training.data.augmentation.augmentation_presets import augmentation_presets_dict

    test_dataset_path = "/home/max/Desktop/Articolo_coigan/COIGAN-IROS-2024/datasets/severstal_steel_defect_dataset/test_IROS2024/tile_train_set"

        # create the augmentor object
    augmentor = Augmentor(
        transforms=augmentation_presets_dict["mask_defects_preset"],
        only_imgs_transforms=augmentation_presets_dict["imgs_defects_preset"]
    )

    dataset = JsonLineDatasetSegm(
        os.path.join(test_dataset_path, "data"),
        os.path.join(test_dataset_path, "dataset.jsonl"),
        os.path.join(test_dataset_path, "index"),
        classes=["0", "1", "2"],
        augmentor=augmentor,
        masks_fields=["polygons"],
        binary=True
    )
    dataset.on_worker_init()

    # visualization

    for i in range(1000):
        sample = dataset[i]
        img = sample["inp"]
        masks = sample["out"]

        # convert image to numpy
        img = (img.permute(1, 2, 0).numpy()*255.0).astype(np.uint8)
        cv2.imshow("img", img)

        for i, _class in enumerate(dataset.classes):
            mask = (masks[i].numpy()*255.0).astype(np.uint8)
            cv2.imshow(f"mask_{_class}", mask)

        # wait key or handle quit
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        