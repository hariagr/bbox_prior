import torch
from collections import OrderedDict
from torchvision.ops import boxes as box_ops
from PIL import Image, ImageDraw
import math
import utils
import time
import datetime
import numpy as np


def cal_bbox_prior_hp(model, data_loader, device):
        bbox_priors = model.bbox_priors

        # define points based on the grid size of the coarsest layer, i.e., the last layer
        # or define points based on the grid size of each feature layer.
        # note that the anchors are of the same size at each center of a cell of a grid
        images, targets = next(iter(data_loader))
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        images, targets = model.transform(images, targets)
        features = model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        features = list(features.values())
        anchors = model.anchor_generator(images, features)

        image_size = images.tensors.shape[-2:]
        center = torch.tensor((image_size[1] / 2, image_size[0] / 2))

        # build stochastic boxes
        n = 2  #model.bbox_prior_coverage
        label = 2
        min_bbstep = 0.05
        max_bbstep = 1

        num_points = 1
        num_classes = data_loader.dataset.num_classes()
        fg_anchors = np.zeros(10, num_points * num_classes)
        for idx, bbp_steps in enumerate(torch.linspace(min_bbstep, max_bbstep, 10, device=device)):
                start_time = time.time()
                xs = torch.linspace(-n, n, int(torch.ceil(torch.tensor(2 * n / bbp_steps))) + 1, device=device)
                ws, hs = torch.meshgrid(xs, xs, indexing="ij")
                for label in range(0, data_loader.dataset.num_classes()):
                        x1 = center[0] - 0.5 * (model.bbox_priors['width_mean'][label] + ws * model.bbox_priors['width_std'][label])
                        x2 = center[0] + 0.5 * (model.bbox_priors['width_mean'][label] + ws * model.bbox_priors['width_std'][label])
                        y1 = center[1] - 0.5 * (model.bbox_priors['height_mean'][label] + hs * model.bbox_priors['height_std'][label])
                        y2 = center[1] + 0.5 * (model.bbox_priors['height_mean'][label] + hs * model.bbox_priors['height_std'][label])

                        stochastic_boxes = torch.stack([x1.flatten(), y1.flatten(), x2.flatten(), y2.flatten()], 1)

                        outer_box = stochastic_boxes[-1, :].reshape(1, -1)
                        all_inner_matched_idx = model.proposal_matcher(utils.box_ioa(outer_box, anchors[0]))

                        all_inner_anchors = anchors[0][all_inner_matched_idx >= 0]
                        quality_matrix = box_ops.box_iou(stochastic_boxes, all_inner_anchors)

                        q2 = torch.max(quality_matrix, 0)[0]
                        qtemp = torch.zeros((1, len(anchors[0])), device=device)
                        qtemp[0, all_inner_matched_idx >= 0] = q2.reshape(1, -1)
                        matched_idx_2 = model.proposal_matcher(qtemp)
                        matched_anchors = anchors[0][matched_idx_2 >= 0]
                        fg_anchors[idx].append(sum(matched_idx_2 >= 0))

                        if 0:
                                img = Image.new("RGB", (image_size[1], image_size[0]))
                                img1 = ImageDraw.Draw(img)
                                for anchor in matched_anchors:
                                        img1.rectangle(anchor.numpy(), outline="blue")
                                img.show()

                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=(total_time)))
                print(f"execution time {total_time_str}")


