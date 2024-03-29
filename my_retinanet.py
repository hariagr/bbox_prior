import math
import warnings
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

import torch
from torch import nn, Tensor

from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.ops import sigmoid_focal_loss
from torchvision.ops import boxes as box_ops
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.utils import _log_api_usage_once
from torchvision.models.resnet import resnet50
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.anchor_utils import AnchorGenerator
from backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from GeneralizedRCNNTransform import GeneralizedRCNNTransform

import utils

__all__ = ["RetinaNet", "retinanet_resnet50_fpn"]


def _sum(x: List[Tensor]) -> Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res


class RetinaNetHead(nn.Module):
    """
    A regression and classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_anchors, num_classes, bl_weights, alpha_ct, alpha, exp_tc, gt_bbox_loss, st_bbox_loss):
        super().__init__()
        self.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes, bl_weights)
        self.regression_head = RetinaNetRegressionHead(in_channels, num_anchors, bl_weights, alpha_ct, alpha, exp_tc, gt_bbox_loss, st_bbox_loss)

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs, epoch):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Dict[str, Tensor]
        return {
            "classification": self.classification_head.compute_loss(targets, head_outputs, matched_idxs),
            "bbox_regression": self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs, epoch),
        }

    def forward(self, x):
        # type: (List[Tensor]) -> Dict[str, Tensor]
        return {"cls_logits": self.classification_head(x), "bbox_regression": self.regression_head(x)}


class RetinaNetClassificationHead(nn.Module):
    """
    A classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_anchors, num_classes, bl_weights, prior_probability=0.01):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.bl_weights  = bl_weights

        # This is to fix using det_utils.Matcher.BETWEEN_THRESHOLDS in TorchScript.
        # TorchScript doesn't support class attributes.
        # https://github.com/pytorch/vision/pull/1697#issuecomment-630255584
        self.BETWEEN_THRESHOLDS = det_utils.Matcher.BETWEEN_THRESHOLDS

    def compute_loss(self, targets, head_outputs, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Tensor
        losses = []
        cls_logits = head_outputs["cls_logits"]

        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):

            # combine labels for deterministic boxes and stochastic boxes
            targets_per_image_label = torch.cat((targets_per_image['labels'], targets_per_image['plabels']), 0)

            # determine only the foreground
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            # create the target classification
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image_label[matched_idxs_per_image[foreground_idxs_per_image]],
            ] = 1.0

            # find indices for which anchors should be ignored
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            # compute the classification loss
            loss_per_image = sigmoid_focal_loss(
                cls_logits_per_image[valid_idxs_per_image],
                gt_classes_target[valid_idxs_per_image],
                reduction='none',
            )
            loss_per_image = loss_per_image * self.bl_weights
            loss_per_image = loss_per_image.sum() / max(1, num_foreground)
            losses.append(loss_per_image)

        return _sum(losses) / len(targets)

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_cls_logits = []

        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, 4)

            all_cls_logits.append(cls_logits)

        return torch.cat(all_cls_logits, dim=1)


class RetinaNetRegressionHead(nn.Module):
    """
    A regression head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
    }

    def __init__(self, in_channels, num_anchors, bl_weights, alpha_ct, alpha, exp_tc, gt_bbox_loss, st_bbox_loss):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.bbox_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.bbox_reg.bias)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.bl_weights = bl_weights
        self.alpha_ct = alpha_ct
        self.alpha = torch.tensor([1, 1, 0, 0], device=bl_weights.device)
        self.cal_tnorm_weights = False
        self.target_normalization = {'x': torch.zeros(4).to(bl_weights.device), 'x2': torch.zeros(4).to(bl_weights.device), 'num': 0}
        self.gt_bbox_loss = gt_bbox_loss
        self.st_bbox_loss = st_bbox_loss
        self.exp_tc = exp_tc

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs, epoch):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Tensor
        losses = []

        bbox_regression = head_outputs["bbox_regression"]

        for targets_per_image, bbox_regression_per_image, anchors_per_image, matched_idxs_per_image in zip(
            targets, bbox_regression, anchors, matched_idxs
        ):
            device = anchors_per_image.device

            # The part of the loss function that is associated with points
            # assumes that the width of stochastic box is equal to mean. Hence, we extend points by mean value
            stochastic_box = torch.zeros((targets_per_image['plabels'].numel(), 4), device=device)
            beta_stbox = torch.zeros((targets_per_image['plabels'].numel(), 4), device=device)
            idx_stbox = torch.zeros((targets_per_image['plabels'].numel(), 1), dtype=torch.int32, device=device)
            if targets_per_image['points'].numel() != 0:
                for indx, (label, center) in enumerate(zip(targets_per_image['plabels'], targets_per_image['points'])):
                    x1 = center[0] - 0.5 * torch.exp(self.bbox_priors['logOfwidth_mean'][label])
                    x2 = center[0] + 0.5 * torch.exp(self.bbox_priors['logOfwidth_mean'][label])
                    y1 = center[1] - 0.5 * torch.exp(self.bbox_priors['logOfheight_mean'][label])
                    y2 = center[1] + 0.5 * torch.exp(self.bbox_priors['logOfheight_mean'][label])
                    stochastic_box[indx] = torch.tensor([x1, y1, x2, y2], device=device).reshape(1, -1)
                    idx_stbox[indx] = label

                    width_mu = self.bbox_priors['width_mean'][label]
                    height_mu = self.bbox_priors['height_mean'][label]
                    sq_area = torch.sqrt(width_mu*height_mu)
                    if not self.cal_tnorm_weights:
                        if self.st_bbox_loss == 'l1':
                            beta_stbox[indx] = torch.tensor(
                                [1 + self.alpha_ct/sq_area, 1 + self.alpha_ct/sq_area, 1, 1], device=device).reshape(1, -1)
                        elif self.st_bbox_loss == 'l2':
                            beta_stbox[indx] = torch.tensor(
                                [1 + (self.alpha_ct**2)/(sq_area**2), 1 + (self.alpha_ct**2)/(sq_area**2), 1, 1], device=device).reshape(1, -1)
                    else:
                        beta_stbox[indx] = torch.tensor([1, 1, 1, 1], device=device).reshape(1, -1)

            beta_box = torch.ones(targets_per_image['boxes'].shape, device=device)
            idx_box = -1*torch.ones(targets_per_image['boxes'].shape[0], dtype=torch.int32, device=device).reshape(-1, 1)

            # remember matched_idx has mapping from anchor to boxes and points
            # entries from 0 to N indicates boxes and N+1 to max(matched_idx) indicates points
            # Hence, we combine deterministic boxes and stochastic boxes array for computing target regression in one call
            boxes_per_image = torch.cat((targets_per_image['boxes'], stochastic_box))
            labels_per_image = torch.cat((targets_per_image['labels'], targets_per_image['plabels']), 0)
            beta_per_image = torch.cat((beta_box, beta_stbox), 0)
            idx_per_image = torch.cat((idx_box, idx_stbox), 0)

            # determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()

            # select only the foreground boxes
            matched_gt_boxes_per_image = boxes_per_image[matched_idxs_per_image[foreground_idxs_per_image]]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
            beta_per_image = beta_per_image[matched_idxs_per_image[foreground_idxs_per_image]]
            idx_per_image = idx_per_image[matched_idxs_per_image[foreground_idxs_per_image]]

            # compute the regression targets
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)

            if self.cal_tnorm_weights:
                self.target_normalization['x'] = self.target_normalization['x'] + torch.sum(target_regression, 0)
                self.target_normalization['x2'] = self.target_normalization['x2'] + torch.sum(torch.pow(target_regression, 2), 0)
                self.target_normalization['num'] = self.target_normalization['num'] + num_foreground
                continue

            # class imbalance weighing
            bl_det_weights = self.bl_weights[labels_per_image[matched_idxs_per_image[foreground_idxs_per_image]]].reshape((-1, 1))

            # compute the loss
            # we know computing both losses are redundant, but it simplifies the code
            if self.gt_bbox_loss == 'l1' or self.gt_bbox_loss == 'l2' or self.st_bbox_loss == 'l1' or self.st_bbox_loss == 'l2':
                det_l1_loss_per_image = torch.nn.functional.l1_loss(bbox_regression_per_image, target_regression,
                                                                    size_average=False, reduce=False, reduction='none')
                if self.gt_bbox_loss == 'l2' or self.st_bbox_loss == 'l2':
                    det_l2_loss_per_image = torch.nn.functional.mse_loss(bbox_regression_per_image, target_regression,
                                                                     size_average=False, reduce=False, reduction='none')
                # considering beta is one for deterministic boxes
                if self.gt_bbox_loss == 'l1':
                    det_loss_per_image = det_l1_loss_per_image
                elif self.gt_bbox_loss == 'l2':
                    det_loss_per_image = det_l2_loss_per_image

                if targets_per_image['points'].numel() != 0:
                    idx_stbox = torch.where(idx_per_image >= 0)[0]
                    #wh = anchors_per_image[idx_stbox, 2:4] - anchors_per_image[idx_stbox, 0:2]
                    alpha = self.alpha
                    if self.st_bbox_loss == 'l1':
                        det_loss_per_image[idx_stbox] = alpha * (1 / beta_per_image[idx_stbox]) * \
                                                        det_l1_loss_per_image[
                                                            idx_stbox]
                    elif self.st_bbox_loss == 'l2':
                        det_loss_per_image[idx_stbox] = alpha * (1 / beta_per_image[idx_stbox]) * \
                                                        det_l2_loss_per_image[
                                                           idx_stbox]

            elif self.gt_bbox_loss == 'smooth_l1' and self.st_bbox_loss == 'smooth_l1':
                det_loss_per_image = torch.zeros(bbox_regression_per_image.shape, device=device)
                uidx = torch.unique(idx_per_image)
                for idx in uidx:
                    idx_box = torch.where(idx_per_image == idx)[0]
                    for col in range(3):
                        det_loss_per_image[idx_box, col] = torch.nn.functional.smooth_l1_loss(bbox_regression_per_image[idx_box, col], target_regression[idx_box, col],
                                                                 size_average=False, reduce=False, reduction='none', beta=beta_per_image[idx_box[0], col])
                    if idx != -1:
                        det_loss_per_image[idx_box] = self.alpha * det_loss_per_image[idx_box]
            else:
                print(f"selected loss function combination is not implemented yet")

            # balancing
            det_loss_per_image = bl_det_weights * det_loss_per_image

            idx_gtbox = torch.where(idx_per_image == -1)[0]
            idx_stbox = torch.where(idx_per_image >= 0)[0]
            gt_loss = sum(sum(det_loss_per_image[idx_gtbox, :])) / max(1, num_foreground - idx_stbox.numel()) if idx_gtbox.numel() != 0 else 0
            st_loss = sum(sum(det_loss_per_image[idx_stbox, :])) / max(1, idx_stbox.numel()) if idx_stbox.numel() != 0 else 0

            loss_per_image = gt_loss + st_loss
            losses.append(loss_per_image)

        if self.cal_tnorm_weights:
            return torch.zeros(0, device=target_regression.device)
        else:
            return _sum(losses) / max(1, len(targets))

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_bbox_regression = []

        for features in x:
            bbox_regression = self.conv(features)
            bbox_regression = self.bbox_reg(bbox_regression)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)

            all_bbox_regression.append(bbox_regression)

        return torch.cat(all_bbox_regression, dim=1)


class RetinaNet(nn.Module):
    """
    Implements RetinaNet.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain an out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or an OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): Module run on top of the feature pyramid.
            Defaults to a module containing a classification and regression module.
        score_thresh (float): Score threshold used for postprocessing the detections.
        nms_thresh (float): NMS threshold used for postprocessing the detections.
        detections_per_img (int): Number of best detections to keep after NMS.
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training.
        topk_candidates (int): Number of best detections to keep before NMS.

    Example:

        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import RetinaNet
        >>> from torchvision.models.detection.anchor_utils import AnchorGenerator
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        >>> # RetinaNet needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the network generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(
        >>>     sizes=((32, 64, 128, 256, 512),),
        >>>     aspect_ratios=((0.5, 1.0, 2.0),)
        >>> )
        >>>
        >>> # put the pieces together inside a RetinaNet model
        >>> model = RetinaNet(backbone,
        >>>                   num_classes=2,
        >>>                   anchor_generator=anchor_generator)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """

    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
    }

    def __init__(
        self,
        backbone,
        num_classes,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # Anchor parameters
        anchor_generator=None,
        head=None,
        proposal_matcher=None,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=300,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.4,
        topk_candidates=1000,
        # loss parameters
        bl_weights=None,
        alpha_ct=1,
        alpha=0,
        exp_tc=0,
        bbox_sampling='mean',
        bbp_coverage=0.0,
        bbp_sampling_step=0.5,
        gt_bbox_loss='l1',
        st_bbox_loss='l2',
        tauc=0.2,
        tauiou=0.2
    ):
        super().__init__()
        _log_api_usage_once(self)

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )
        self.backbone = backbone

        assert isinstance(anchor_generator, (AnchorGenerator, type(None)))

        if anchor_generator is None:
            anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128])  #, 256, 512
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        self.anchor_generator = anchor_generator

        if head is None:
            head = RetinaNetHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes, bl_weights, alpha_ct, alpha, exp_tc, gt_bbox_loss, st_bbox_loss)
        self.head = head

        if proposal_matcher is None:
            proposal_matcher = det_utils.Matcher(
                fg_iou_thresh,
                bg_iou_thresh,
                allow_low_quality_matches=True,
            )
        self.proposal_matcher = proposal_matcher

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates

        # used only on torchscript mode
        self._has_warned = False

        # bounding box prior
        #self.box_priors = box_priors
        self.bbox_sampling = bbox_sampling
        self.bbox_prior_coverage = bbp_coverage
        self.bbox_prior_sampling_step = bbp_sampling_step
        self.tauc = tauc
        self.tauiou = tauiou

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def compute_loss(self, targets, head_outputs, anchors):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Dict[str, Tensor]

        bbox_regression = head_outputs['bbox_regression'].detach()
        cls_scores = torch.sigmoid(head_outputs["cls_logits"].detach())

        matched_idxs = []
        for img_idx, (anchors_per_image, targets_per_image, bbox_regression_per_image, cls_scores_per_image) in enumerate(zip(anchors, targets, bbox_regression, cls_scores)):
            device = anchors_per_image.device

            if targets_per_image["boxes"].numel() == 0 and targets_per_image['points'].numel() == 0 and \
                    targets_per_image["gboxes"].numel() == 0 and targets_per_image["mboxes"].numel() == 0:
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=device)  # set all anchors as background
                )
                continue

            if targets_per_image["boxes"].numel() != 0:
                match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
                bmatched_idxs = self.proposal_matcher(match_quality_matrix)
                bmatched_anchors = bmatched_idxs >= 0
            else:
                match_quality_matrix = torch.tensor([], device=device)
                bmatched_idxs = torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=device)

            if targets_per_image['points'].numel() != 0:

                pred_boxes = self.box_coder.decode_single(bbox_regression_per_image, anchors_per_image)
                st_boxes = torch.zeros(targets_per_image['points'].shape[0], 4, device=device)
                for idx, (label, center) in enumerate(zip(targets_per_image['plabels'], targets_per_image['points'])):

                    if self.bbox_sampling == 'mean':
                        width_mu = self.bbox_priors['width_mean'][label]
                        height_mu = self.bbox_priors['height_mean'][label]
                    elif self.bbox_sampling == 'mode':
                        width_mu = self.bbox_priors['width_mode'][label]
                        height_mu = self.bbox_priors['height_mode'][label]
                    elif self.bbox_sampling == 'mean_IOU':
                        width_mu = self.bbox_priors['width_mean_IOU'][label]
                        height_mu = self.bbox_priors['height_mean_IOU'][label]
                    else:
                        print(f"{self.bbox_sampling} sampling method is not implemented!")

                    x1 = center[0] - 0.5 * width_mu
                    x2 = center[0] + 0.5 * width_mu
                    y1 = center[1] - 0.5 * height_mu
                    y2 = center[1] + 0.5 * height_mu

                    # create all stochastic boxes
                    stochastic_boxes = torch.stack([x1.flatten(), y1.flatten(), x2.flatten(), y2.flatten()], 1)
                    st_boxes[idx, :] = stochastic_boxes

                    # filtering: choose a fixed stochastic box or anchor as a stochastic box
                    # IOU between predicted box and stochastic box
                    iou = box_ops.box_iou(st_boxes[idx, :].reshape(1, -1), pred_boxes)
                    score = cls_scores_per_image[:, label].reshape(1, -1)
                    if self.tauiou == -1:
                        tauiou = self.bbox_priors['mean_IOU'][label]
                    else:
                        tauiou = self.tauiou
                    credible_box_indx = torch.where((iou >= tauiou) & (score >= self.tauc))[1]
                    if credible_box_indx.numel() > 0:
                        sel_boxes = pred_boxes[credible_box_indx, :]
                        score = score[0, credible_box_indx]
                        pred_center = sel_boxes[:, 0:2] + 0.5*(sel_boxes[:, 2:4] - sel_boxes[:, 0:2])
                        distance = torch.sum((pred_center - center) ** 2, 1)
                        sel_boxes_index = torch.argmax(torch.exp(-distance)*score)
                        st_boxes[idx, :] = sel_boxes[sel_boxes_index, :]

                targets[img_idx]['st_boxes'] = st_boxes     # store st boxes for box regression
                pmatch_quality_matrix = box_ops.box_iou(st_boxes, anchors_per_image)
                match_quality_matrix = torch.cat((match_quality_matrix, pmatch_quality_matrix))
                bmatched_idxs = self.proposal_matcher(match_quality_matrix)
            else:  # clusters are considered only if point information is not available
                if targets_per_image["gboxes"].numel() != 0:
                    gmatch_quality_matrix = utils.box_ioa(targets_per_image['gboxes'],
                                                          anchors_per_image)  # anchors lying inside the gboxes
                    gmatched_idxs = self.proposal_matcher(gmatch_quality_matrix)
                    gmatched_anchors = gmatched_idxs >= 0
                    # single cell anchors lying inside clusters are removed from matching vector
                    gmatched_anchors[bmatched_anchors] = False
                    bmatched_idxs[gmatched_anchors] = -2  # BETWEEN_THRESHOLDS, see proposal_matcher function

            if targets_per_image['mboxes'].numel() != 0:
                # IOU or IOA (decide) for missed labels
                mmatch_quality_matrix = box_ops.box_iou(targets_per_image['mboxes'], anchors_per_image)  # anchors lying inside the gboxes
                mmatched_idxs = self.proposal_matcher(mmatch_quality_matrix)
                mmatched_anchors = mmatched_idxs >= 0
                bmatched_idxs[mmatched_anchors] = -2

            matched_idxs.append(bmatched_idxs)

        return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs, self.epoch)

    def postprocess_detections(self, head_outputs, anchors, image_shapes):
        # type: (Dict[str, List[Tensor]], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]

        num_images = len(image_shapes)

        detections: List[Dict[str, Tensor]] = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            anchors_per_image, image_shape = anchors[index], image_shapes[index]

            image_boxes = []
            image_scores = []
            image_labels = []

            for box_regression_per_level, logits_per_level, anchors_per_level in zip(
                box_regression_per_image, logits_per_image, anchors_per_image
            ):
                num_classes = logits_per_level.shape[-1]

                # remove low scoring boxes
                scores_per_level = torch.sigmoid(logits_per_level).flatten()
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                num_topk = min(self.topk_candidates, topk_idxs.size(0))
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode_single(
                    box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
                )
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            if 0:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                            raise ValueError(f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.")
                    else:
                        raise ValueError(f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if 0:
            if targets is not None:
                for target_idx, target in enumerate(targets):
                    boxes = target["boxes"]
                    degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                    if degenerate_boxes.any():
                        # print the first degenerate box
                        bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                        degen_bb: List[float] = boxes[bb_idx].tolist()
                        raise ValueError(
                            "All bounding boxes should have positive height and width."
                            f" Found invalid box {degen_bb} for target at index {target_idx}."
                        )
        # get the features from the backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # TODO: Do we want a list or a dict?
        features = list(features.values())

        # compute the retinanet heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            assert targets is not None

            # compute the losses
            losses = self.compute_loss(targets, head_outputs, anchors)
        else:
            # recover level sizes
            num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
            HW = 0
            for v in num_anchors_per_level:
                HW += v
            HWA = head_outputs["cls_logits"].size(1)
            A = HWA // HW
            num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

            # split outputs per level
            split_head_outputs: Dict[str, List[Tensor]] = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            # compute the detections
            detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RetinaNet always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        return self.eager_outputs(losses, detections)


model_urls = {
    "retinanet_resnet50_fpn_coco": "https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth",
}


def retinanet_resnet50_fpn(
    pretrained=False, progress=True, num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None, freeze_bn=False, **kwargs
):
    """
    Constructs a RetinaNet model with a ResNet-50-FPN backbone.

    Reference: `"Focal Loss for Dense Object Detection" <https://arxiv.org/abs/1708.02002>`_.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detections:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each detection
        - scores (``Tensor[N]``): the scores of each detection

    For more details on the output, you may refer to :ref:`instance_seg_output`.

    Example::

        >>> model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable. If ``None`` is
            passed (the default) this value is set to 3.
    """
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3
    )

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False

    norm_layer = None
    if freeze_bn:
        norm_layer = misc_nn_ops.FrozenBatchNorm2d
    #print(norm_layer)

    backbone = resnet50(pretrained=pretrained_backbone, progress=progress, norm_layer=norm_layer)
    #backbone = resnet50(pretrained=pretrained_backbone, progress=progress)
    # skip P2 because it generates too many anchors (according to their paper)
    backbone = _resnet_fpn_extractor(
        backbone, trainable_backbone_layers, returned_layers=[2, 3, 4], extra_blocks=None  #LastLevelP6P7(256, 256)
    )
    model = RetinaNet(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["retinanet_resnet50_fpn_coco"], progress=progress)
        model.load_state_dict(state_dict)
        overwrite_eps(model, 0.0)
    return model
