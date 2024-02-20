import os
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from omegaconf import OmegaConf
from tqdm import tqdm

from COIGAN.modules import make_segmentation_model
from COIGAN.utils.common_utils import make_optimizer, make_lr_scheduler
from COIGAN.training.logger import DataLogger
from COIGAN.segmentation.losses import loss_mng
from COIGAN.segmentation.losses import dice_loss, log_cos_dice

LOGGER = logging.getLogger(__name__)

output_classes = ['0', '1', '2']
input_classes = ['no_damage', 'damage']
losses = {
        "log_cos_dice": log_cos_dice(),
        #"dice": dice_loss(),
        #"bce_with_logits": bce_logit_loss(on_active=True, min_threshold=0.05),
        #"border_loss": border_loss()
        #"bce_logit": bce_logit_loss()
    }
loss_mng_train_conf = {
    "losses": losses,
    "classes": output_classes,
    "input_classes": input_classes,
    "loss_weights": None, #[2.0, 2.0], # None means equal weights, applyed if the class_loss_weight is not defined for a class
    "classes_weights": None, #[0.3, 1.2, 1.2, 1.1, 1.2], # None means equal weights
    "input_classes_weights": None #[1.0, 1.0], # None means equal weights
}
loss_mng_val_conf = {
    "losses": losses, #{"dice": dice_loss()},
    "classes": output_classes,
    "input_classes": input_classes,
    "loss_weights": None, #[1.0], # None means equal weights, applyed if the class_loss_weight is not defined for a class
    "classes_weights": None, #[1.0, 1.0, 1.0, 1.0, 1.0], # None means equal weights
    "input_classes_weights": None, #[1.0, 1.0], # None means equal weights
}

class SegmentationTrainer:

    def __init__(
            self, 
            config: OmegaConf, 
            dataloader: DataLoader,
            val_dataloader: DataLoader
        ):
        
        # training variables
        self.global_step = 0

        # config
        self.config = config

        # dataloaders
        self.dataloader = dataloader # training dataloader
        self.val_dataloader = val_dataloader # validation dataloader

        # training parameters
        self.batch_size = self.config.batch_size # batch size
        self.epochs = self.config.epochs # number of epochs to train the model
        self.val_interval = self.config.val_interval # number of epochs between each validation

        # logging and checkpointing parameters
        self.log_img_interval = self.config.log_img_interval # number of epochs between each image logging
        self.log_weights_interval = self.config.log_weights_interval # number of epochs between each weights logging
        self.checkpoint_interval = self.config.checkpoint_interval # number of epochs between each checkpoint saving

        # number of samples
        self.n_train = len(dataloader.dataset) # number of training samples
        self.n_val = len(val_dataloader.dataset) # number of validation samples
        self.n_train_batches = self.n_train // self.batch_size # number of training batches
        self.n_val_batches = self.n_val // self.batch_size # number of validation batches
        
        # device and amp
        self.amp = self.config.amp # if True, use the automatic mixed precision
        self.device = self.config.device # device to train the model on

        # paths
        self.checkpoint = self.config.checkpoint # path to the checkpoint to load
        self.checkpoints_path = self.config.location.checkpoint_dir # path to save the checkpoints

        # model, optimizer and learning rate scheduler
        self.model = make_segmentation_model(**config.model).to(self.device)
        self.optim = make_optimizer(self.model, **config.optimizers.model)
        self.scheduler = make_lr_scheduler(self.optim, **config.optimizers.lr_scheduler)
        
        # learning rate scheduler ! (Not integrated in specific function for time reasons)
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #    self.optim,
        #    "max",
        #    patience=10, # number of lr_scheduler updates (with a static treand) to wait before decaying the learning rate
        #    factor=0.95, # factor to decay the learning rate each period of static loss
        #    min_lr=1e-6
        #)  # goal: maximize Dice score


        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        # creation of the loss managers
        self.loss_manager = loss_mng(loss_mng_train_conf)
        self.val_loss_manager = loss_mng(loss_mng_val_conf, eval=True)

        # inizialization of the datalogger
        self.datalogger = DataLogger(
                **self.config.logger,
                config=self.config
            )

        # loading checkpoint
        if self.checkpoint is not None:
            self.load_checkpoint(self.checkpoint)


    def load_checkpoint(self, checkpoint_file_path):
        
        checkpoint = torch.load(checkpoint_file_path)
        LOGGER.info(f"Loading checkpoint: {checkpoint_file_path}")

        self.model.load_state_dict(checkpoint["model"])
        self.optim.load_state_dict(checkpoint["optim"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

        # extract the epoch from the checkpoint path
        epoch = int(os.path.basename(checkpoint_file_path).split(".")[0])
        self.global_step = epoch * self.n_train_batches


    def save_checkpoint(self, epoch):

        checkpoint_path = os.path.join(self.checkpoints_path, f"{epoch}.pt")
        LOGGER.info(f"Saving checkpoint: {checkpoint_path}")

        torch.save({
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }, checkpoint_path)


    def evaluate(self):

        # reset the validation loss manager, to handle the new validation round
        self.val_loss_manager.reset_val()

        # set the model to evaluation mode
        self.model.eval()

        # validation loop
        with tqdm(total=self.n_val, desc=f"Validation round", unit="imgs", leave=False) as pbar:
            for batch in self.val_dataloader:
                imgs = batch["inp"].to(self.device) # input images
                masks = batch["out"].to(self.device) # target masks (ground truth)
                in_class = batch["fill"] # fill value for the masks (used to classify the input classes [No damage, Damage])

                with torch.no_grad():
                    mask_pred = self.model(imgs)
                    mask_pred = torch.sigmoid(mask_pred)
                    self.val_loss_manager(
                        (mask_pred > 0.5).float(), 
                        masks, 
                        in_class
                    )

                pbar.update(self.batch_size) # update the progress bar, by the batch size
        
        # calculating the val loss across all the validation set
        val_loss = self.val_loss_manager.get_val_loss()
        LOGGER.info(f"Validation loss: {val_loss}")

        # update the learning rate scheduler
        self.scheduler.step(val_loss)

        # log the results
        self.datalogger.log_step_results(
            self.global_step, 
            {
                "val_loss": val_loss,
                "val_input_classwise_losses": self.val_loss_manager.get_val_losses_input_classwise(),
                "val_classwise_losses": self.val_loss_manager.get_val_losses_classwise(),
                "val_losswise_losses": self.val_loss_manager.get_val_losses_losswise()
            }
        )

        self.model.train()


    def train(self):

        # set the model to train mode
        self.model.train()

        # training main loop
        for epoch in range(self.epochs):
            with tqdm(total=self.n_train, desc=f"Epoch {epoch}/{self.epochs}", unit="imgs") as pbar:
                for batch in self.dataloader:
                    imgs = batch["inp"].to(self.device) # input images
                    masks = batch["out"].to(self.device) # target masks (ground truth)
                    in_class = batch["fill"] # fill value for the masks (used to classify the input classes [No damage, Damage])

                    with torch.cuda.amp.autocast(enabled=self.amp):
                        output = self.model(imgs)
                        loss = self.loss_manager(output, masks, in_class)
                    
                    self.optim.zero_grad()
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.step(self.optim)
                    self.grad_scaler.update()

                    pbar.update(self.batch_size) # update the progress bar, by the batch size
                    pbar.set_postfix(**{"loss (batch)": loss.item()})

                    self.global_step += 1

                    # log the results
                    self.datalogger.log_step_results(
                        self.global_step, 
                        {
                            "train_loss": loss.item(),
                            "input_classwise_losses": self.loss_manager.get_losses_input_classwise(),
                            "classwise_losses": self.loss_manager.get_losses_classwise(),
                            "losswise_losses": self.loss_manager.get_losses_losswise()
                        }
                    )

                    # validation
                    if self.global_step % self.val_interval == 0:
                        self.evaluate()
                        self.save_checkpoint(epoch, self.global_step)
                    

                    ##############################################################################################
                    # logging, checkpointing and visualization ###################################################

                    # logging images
                    if self.global_step % self.log_img_interval == 0:
                        self.datalogger.log_visual_results(
                            self.global_step, 
                            {
                                "input": self.make_grid(imgs),
                                "output": self.make_grid(output),
                                "target": self.make_grid(torch.sigmoid(masks))
                            }
                        )
                    
                    # logging weights
                    # NOTE: the weights logging is internally scheduled by the datalogger
                    self.datalogger.log_weights_and_gradients(self.global_step, self.model)


    def make_grid(self, sample):
        """
        Method that wrap the torch make_grid function to create a grid of images.
        with predefined parameters.

        Args:
            sample (torch.Tensor): a tensor containing the images to be plotted (shape: [batch_size, 3, H, W] or [batch_size, 1,  H, W])
        
        Returns:
            torch.Tensor: a tensor containing the grid of images (shape: [3, H, W] or [1, H, W])
        """
        
        return make_grid(
            sample,
            nrow= int(np.sqrt(sample.shape[0])),
            normalize=True,
            range=(0, 1)
        )