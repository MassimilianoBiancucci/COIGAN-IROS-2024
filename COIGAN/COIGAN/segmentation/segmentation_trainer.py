import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf


from COIGAN.modules import make_segmentation_model
from COIGAN.utils.common_utils import make_optimizer


class SegmentationTrainer:

    def __init__(self, config: OmegaConf, dataloader: DataLoader):
        self.config = config
        self.dataloader = dataloader

        self.device = self.config.device

        self.model = make_segmentation_model(**config.model.kwargs).to(self.device)

        self.optim = make_optimizer(self.model, **config.optimizer.model)

        

    def train(self):
        # This is the main training loop
        for epoch in range(self.config.training.epochs):
            for batch in self.dataloader:
                # do something with the batch
                pass