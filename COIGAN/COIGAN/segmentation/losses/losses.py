import torch
from torch.functional import F
import numpy as np
import cv2

class dice_loss:

    std_conf = {
        "epsilon": 1e-6,
        "applay_sigmoid": True
    }

    def __init__(self, config=None):
        
        if config is None:
            self.config = self.std_conf
        
        self.epsilon = self.config["epsilon"]
        self.applay_sigmoid = self.config["applay_sigmoid"]


    def __call__(self, input, target):
        assert input.size() == target.size()

        if input.dim() != 2:
            raise ValueError("dice_loss: input must be 2D!")
        
        # Applay the sigmoid following the configuration
        if self.applay_sigmoid:
            input = torch.sigmoid(input)

        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return 1 - ((2 * inter + self.epsilon)/ (sets_sum + self.epsilon))
    

    def __repr__(self):
        return "dice_loss"


class log_cos_dice:

    std_conf = {
        "epsilon": 1e-6,
        "applay_sigmoid": True
    }

    def __init__(self, config=None):
        
        if config is None:
            self.config = self.std_conf
        
        self.epsilon = self.config["epsilon"]
        self.applay_sigmoid = self.config["applay_sigmoid"]


    def __call__(self, input, target):
        assert input.size() == target.size()

        if input.dim() != 2:
            raise ValueError("dice_loss: input must be 2D!")
        
        # Applay the sigmoid following the configuration
        if self.applay_sigmoid:
            input = torch.sigmoid(input)

        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        dice_loss = 1 - ((2 * inter + self.epsilon)/ (sets_sum + self.epsilon))
        return 2.3*torch.log(torch.cosh(dice_loss))

    def __repr__(self):
        return "log_cos_dice_loss"


class bce_loss:

    def __init__(self, config=None):
        self.loss = torch.nn.BCELoss()

    def __call__(self, input, target):

        return self.loss(input, target)
    
    def __repr__(self):
        return "bce_loss"


class bce_logit_loss:

    def __init__(self, config=None, on_active=False, min_threshold=0.2):
        """
        Args:
            config: A dictionary with the following keys:
            
            on_active: A boolean that indicates if the loss should be applied only on the active pixels.
        """
        self.on_active = on_active
        self.min_threshold = min_threshold
        if not self.on_active:
            self.loss = torch.nn.BCEWithLogitsLoss()

    def __call__(self, input, target):
        """
        Args:
            input: A tensor with the shape (height, width) normalized between 0 and 1.
            target: A tensor with the shape (height, width) normalized between 0 and 1.
        """
        if self.on_active:
            # create a tenor with the same shape as the input and target
            # and fill it with ones where the target is 1 or the input is higher than a minium threshold
            mask = torch.zeros_like(input)
            mask[target == 1] = 1
            mask[input > self.min_threshold] = 1
            # apply the weights to the loss
            self.loss = torch.nn.BCEWithLogitsLoss(weight=mask, reduction="sum")
            raw_loss = self.loss(input, target)
            sum_ = torch.sum(mask)+1e-6
            result = raw_loss / sum_
            return result
        
        else:
            return self.loss(input, target)
    
    def __repr__(self):
        return "bce_logit_loss"


class border_loss:

    """
    Binary cross entropy that apply a weight mask
    on the function based on the boundraise of the target mask.
    """

    std_conf = {
        "dilate_coeff": 100, # the coefficient for the dilation kernel size, the bigger the coeff the smaller the kernel size
        "dilate_k": (3, 3),
        "dilate_it": 1,
        "smooth_k": (11, 11),
        "smooth_sigma": 11
    }

    def __init__(self, config=None):
        
        if config is None:
            self.config = self.std_conf

        # dilation params
        self.dilate_coeff = self.config["dilate_coeff"]
        self.dilate_k = self.config["dilate_k"]
        self.dilate_it = self.config["dilate_it"]

        # smoothing params
        self.smooth_k = self.config["smooth_k"]
        self.smooth_sigma = self.config["smooth_sigma"]

        self.loss = None


    def __call__(self, input, target):
        
        # calculate the weight mask
        weights_mask = self._compute_weight_mask(target)
        weights_mask = torch.from_numpy(weights_mask).cuda() # note cuda is needed to be the same device as the input

        # compute the BCEloss with the weight mask
        self.loss = torch.nn.BCEWithLogitsLoss(weight=weights_mask, reduction="sum")

        return self.loss(input, target)/(weights_mask.sum()+1.0)


    def _compute_weight_mask(self, target):
        """
        Compute the weight mask based on the target mask.
        
        Args:
            target (torch.Tensor): The target mask. with shape (H, W)
        
        Returns:
            np.array (float32): The weight mask.
        """

        # restore from the tensor to the numpy array
        # and bring the tensor from the range [0, 1] to [0, 255]
        target = target.cpu().numpy()
        target = (target * 255).astype(np.uint8)

        # compute the borders of the target mask
        target_border = cv2.Canny(target, 0, 1)

        # adjust the borders dilation kernel for the target shape
        coeff = (sum(target.shape)/self.dilate_coeff)
        dilate_kernel = (int(self.dilate_k[0] * coeff)+1, int(self.dilate_k[1] * coeff)+1)
        dilate_kernel = np.ones(dilate_kernel, np.uint8)

        # enlarge the borders with a dilate
        target_border = cv2.dilate(target_border, dilate_kernel, iterations=self.dilate_it)
        # apply a smoothing filter
        target_border = cv2.GaussianBlur(target_border, self.smooth_k, self.smooth_sigma)

        # restore the borders to the range [0, 1]
        target_border = target_border.astype(np.float32)/255

        # rescale the border tensor to the range [0, 1] from [0, max] 
        # (max should be less than 1)
        max_val = np.max(target_border)
        if max_val > 0:
            target_border *= 1/target_border.max()
        else:
            target_border = np.zeros_like(target_border)

        return target_border

    

    def __repr__(self):
        return "shape_loss"


class pos_points_loss:

    """
    Loss function that penalize much more if the model predicts 
    incorrectly where there are positive points.
    """

    std_conf = {

    }

    def __init__(self, config=None):
        
        if config is None:
            self.config = self.std_conf
        
    def __call__(self, input, output, target):
        """
        Args:
            input: the input of the model
                es. (channels, height, width)
                channels -> positive points, negative points, image

            output: the output of the model
                es. (height, width)

            target: the target of the model
                es. (height, width)
        """

        assert output.size() == target.size()
        

if __name__ == "__main__":
    

    for i in range(10):

        j = i + 1
        h = j*100
        w = j*100
        c_x = j*50
        c_y = j*50
        d_c = j*5
        r = j*20

        # generate a target mask with a circle
        target_mask = np.zeros((h, w), np.uint8)
        cv2.circle(target_mask, (c_y, c_x), r, 1, -1)
        target_mask = torch.from_numpy(target_mask).float()

        # generate a pred mask with a circle with a little random shift
        pred_mask = np.zeros((h, w), np.uint8)
        cv2.circle(pred_mask, (c_y + np.random.randint(-d_c, d_c), c_x + np.random.randint(-d_c, d_c)), r, 1, -1)
        pred_mask = torch.from_numpy(pred_mask/255).float()

        conf = {

        }

        loss = border_loss()

        weights_mask = loss._compute_weight_mask(target_mask)

        # show the target and the weight mask

        target_mask = target_mask.numpy()
        target_mask = (target_mask*255).astype(np.uint8)
        cv2.imshow("target", target_mask)


        cv2.imshow("weights", weights_mask)

        cv2.waitKey(0) 