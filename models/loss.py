import torch.nn.functional as F
import torch
def isNaN(x):
    return x != x

def model_loss(disp_ests, disp_gt, mask):
    weights1 = [0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.1]
    weights2 = [1.8, 2.1]
    assert len(weights2) == len(disp_ests)
    weights = weights2
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)

def focal_loss(disp_ests, disp_gt, maxdisp, focal_coefficient, sparse):
    focal_loss_evaluator = \
        StereoFocalLoss(max_disp=maxdisp, focal_coefficient=focal_coefficient, sparse=sparse)

    weights = [0.5, 0.7, 1.0, 1.2, 1.5]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * focal_loss_evaluator(disp_est, disp_gt, variance=1))
    return sum(all_losses)

class Disp2Prob(object):
    """
    Convert disparity map to matching probability volume
        Args:
            maxDisp, (int): the maximum of disparity
            gtDisp, (torch.Tensor): in (..., Height, Width) layout
            start_disp (int): the start searching disparity index, usually be 0
            dilation (int): the step between near disparity index

        Outputs:
            probability, (torch.Tensor): in [BatchSize, maxDisp, Height, Width] layout


    """
    def __init__(self, maxDisp, gtDisp, start_disp=0, dilation=1):

        if not isinstance(maxDisp, int):
            raise TypeError('int is expected, got {}'.format(type(maxDisp)))

        if not torch.is_tensor(gtDisp):
            raise TypeError('tensor is expected, got {}'.format(type(gtDisp)))

        if not isinstance(start_disp, int):
            raise TypeError('int is expected, got {}'.format(type(start_disp)))

        if not isinstance(dilation, int):
            raise TypeError('int is expected, got {}'.format(type(dilation)))

        if gtDisp.dim() == 2:  # single image H x W
            gtDisp = gtDisp.view(1, 1, gtDisp.size(0), gtDisp.size(1))

        if gtDisp.dim() == 3:  # multi image B x H x W
            gtDisp = gtDisp.view(gtDisp.size(0), 1, gtDisp.size(1), gtDisp.size(2))

        if gtDisp.dim() == 4:
            if gtDisp.size(1) == 1:  # mult image B x 1 x H x W
                gtDisp = gtDisp
            else:
                raise ValueError('2nd dimension size should be 1, got {}'.format(gtDisp.size(1)))

        self.gtDisp = gtDisp
        self.maxDisp = maxDisp
        self.start_disp = start_disp
        self.dilation = dilation
        self.end_disp = start_disp + maxDisp - 1
        self.disp_sample_number = (maxDisp + dilation -1) // dilation
        self.eps = 1e-40

    def getProb(self):
        # [BatchSize, 1, Height, Width]
        b, c, h, w = self.gtDisp.shape
        assert c == 1

        # if start_disp = 0, dilation = 1, then generate disparity candidates as [0, 1, 2, ... , maxDisp-1]
        self.index = torch.arange(0, self.maxDisp, dtype=self.gtDisp.dtype, device=self.gtDisp.device)
        self.index = self.index.view(1, self.maxDisp, 1, 1)

        # [BatchSize, maxDisp, Height, Width]
        self.index = self.index.repeat(b, 1, h, w).contiguous()

        # the gtDisp must be (start_disp, end_disp), otherwise, we have to mask it out
        mask = (self.gtDisp > self.start_disp) & (self.gtDisp < self.end_disp)
        mask = mask.detach().type_as(self.gtDisp)
        self.gtDisp = self.gtDisp * mask

        probability = self.calProb()

        # let the outliers' probability to be 0
        # in case divide or log 0, we plus a tiny constant value
        probability = probability * mask + self.eps

        # in case probability is NaN
        if isNaN(probability.min()) or isNaN(probability.max()):
            print('Probability ==> min: {}, max: {}'.format(probability.min(), probability.max()))
            print('Disparity Ground Truth after mask out ==> min: {}, max: {}'.format(self.gtDisp.min(),
                                                                                      self.gtDisp.max()))
            raise ValueError(" \'probability contains NaN!")

        return probability

    def kick_invalid_half(self):
        distance = self.gtDisp - self.index
        invalid_index = distance < 0
        # after softmax, the valid index with value 1e6 will approximately get 0
        distance[invalid_index] = 1e6
        return distance

    def calProb(self):
        raise NotImplementedError


class LaplaceDisp2Prob(Disp2Prob):
    # variance is the diversity of the Laplace distribution
    def __init__(self, maxDisp, gtDisp, variance=1, start_disp=0, dilation=1):
        super(LaplaceDisp2Prob, self).__init__(maxDisp, gtDisp, start_disp, dilation)
        self.variance = variance

    def calProb(self):
        # 1/N * exp( - (d - d{gt}) / var), N is normalization factor, [BatchSize, maxDisp, Height, Width]
        scaled_distance = ((-torch.abs(self.index - self.gtDisp)))
        probability = F.softmax(scaled_distance, dim=1)

        return probability

class GaussianDisp2Prob(Disp2Prob):
    # variance is the variance of the Gaussian distribution
    def __init__(self, maxDisp, gtDisp, variance=1, start_disp=0, dilation=1):
        super(GaussianDisp2Prob, self).__init__(maxDisp, gtDisp, start_disp, dilation)
        self.variance = variance

    def calProb(self):
        # 1/N * exp( - (d - d{gt})^2 / b), N is normalization factor, [BatchSize, maxDisp, Height, Width]
        distance = (torch.abs(self.index - self.gtDisp))
        scaled_distance = (- distance.pow(2.0) / self.variance)
        probability = F.softmax(scaled_distance, dim=1)

        return probability

class OneHotDisp2Prob(Disp2Prob):
    # variance is the variance of the OneHot distribution
    def __init__(self, maxDisp, gtDisp, variance=0.5001, start_disp=0, dilation=1):
        super(OneHotDisp2Prob, self).__init__(maxDisp, gtDisp, start_disp, dilation)
        self.variance = variance

    def getProb(self):
        # |d - d{gt}| < variance, [BatchSize, maxDisp, Height, Width]
        b, c, h, w = self.gtDisp.shape
        assert c == 1

        # if start_disp = 0, dilation = 1, then generate disparity candidates as [0, 1, 2, ... , maxDisp-1]
        self.index = torch.arange(0, self.maxDisp, dtype=self.gtDisp.dtype, device=self.gtDisp.device)
        self.index = self.index.view(1, self.maxDisp, 1, 1)

        # [BatchSize, maxDisp, Height, Width]
        self.index = self.index.repeat(b, 1, h, w).contiguous()
        probability = torch.lt(torch.abs(self.index - self.gtDisp), self.variance).type_as(self.gtDisp)

        return probability




class StereoFocalLoss(object):
    """
    Under the same start disparity and maximum disparity, calculating all estimated cost volumes' loss
        Args:
            max_disp, (int): the max of Disparity. default: 192
            start_disp, (int): the start searching disparity index, usually be 0
            dilation (int): the step between near disparity index, it mainly used in gt probability volume generation
            weights, (list of float or None): weight for each scale of estCost.
            focal_coefficient, (float): stereo focal loss coefficient, details please refer to paper. default: 0.0
            sparse, (bool): whether the ground-truth disparity is sparse, for example, KITTI is sparse, but SceneFlow is not. default: False

        Inputs:
            estCost, (Tensor or list of Tensor): the estimated cost volume, in (BatchSize, max_disp, Height, Width) layout
            gtDisp, (Tensor): the ground truth disparity map, in (BatchSize, 1, Height, Width) layout.
            variance, (Tensor or list of Tensor): the variance of distribution, details please refer to paper, in (BatchSize, 1, Height, Width) layout.

        Outputs:
            loss, (dict), the loss of each level

        ..Note:
            Before calculate loss, the estCost shouldn't be normalized,
              because we will use softmax for normalization
    """

    def __init__(self, max_disp=192, start_disp=0, dilation=1, weights=None, focal_coefficient=0.0, sparse=False):
        self.max_disp = max_disp
        self.start_disp = start_disp
        self.dilation = dilation
        self.weights = weights
        self.focal_coefficient = focal_coefficient
        self.sparse = sparse
        if sparse:
            # sparse disparity ==> max_pooling
            self.scale_func = F.adaptive_max_pool2d
        else:
            # dense disparity ==> avg_pooling
            self.scale_func = F.adaptive_avg_pool2d

    def loss_per_level(self, estCost, gtDisp, variance=1.0, dilation=1):
        N, C, H, W = estCost.shape
        scaled_gtDisp = gtDisp.clone()
        scale = 1.0
        if gtDisp.shape[-2] != H or gtDisp.shape[-1] != W:
            # compute scale per level and scale gtDisp
            scale = gtDisp.shape[-1] / (W * 1.0)
            scaled_gtDisp = gtDisp.clone() / scale

            scaled_gtDisp = self.scale_func(scaled_gtDisp, (H, W))

        # mask for valid disparity
        # (start_disp, max disparity / scale)
        # Attention: the invalid disparity of KITTI is set as 0, be sure to mask it out
        lower_bound = self.start_disp
        upper_bound = lower_bound + int(self.max_disp/scale)
        mask = (scaled_gtDisp > lower_bound) & (scaled_gtDisp < upper_bound)
        mask = mask.detach_().type_as(scaled_gtDisp)
        if mask.sum() < 1.0:
            print('Stereo focal loss: there is no point\'s '
                  'disparity is in [{},{})!'.format(lower_bound, upper_bound))
            scaled_gtProb = torch.zeros_like(estCost)  # let this sample have loss with 0
        else:
            # transfer disparity map to probability map
            mask_scaled_gtDisp = scaled_gtDisp * mask
            scaled_gtProb = LaplaceDisp2Prob(int(self.max_disp/scale), mask_scaled_gtDisp, variance=variance,
                                             start_disp=self.start_disp, dilation=dilation).getProb()


        # stereo focal loss
        estProb = F.log_softmax(estCost, dim=1)
        weight = (1.0 - scaled_gtProb).pow(-self.focal_coefficient).type_as(scaled_gtProb)
        loss = -((scaled_gtProb * estProb) * weight * mask.float()).sum(dim=1, keepdim=True).mean()

        return loss

    def __call__(self, estCost, gtDisp, variance):
        # compute loss for per level
        loss_all_level = []
        loss_all_level = self.loss_per_level(estCost, gtDisp, variance)

        return loss_all_level

