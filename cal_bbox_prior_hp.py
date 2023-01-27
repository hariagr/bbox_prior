import torch
from collections import OrderedDict
from torchvision.ops import boxes as box_ops
from PIL import Image, ImageDraw
import math
import utils
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def cal_bbox_prior_hp(model, dataloader, device):
        bbox_priors = model.bbox_priors

        # define points based on the grid size of the coarsest layer, i.e., the last layer
        # or define points based on the grid size of each feature layer.
        # note that the anchors are of the same size at each center of a cell of a grid
        for images, targets in dataloader:
                #images, targets = next(iter(dataloader))
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                images, targets = model.transform(images, targets)
                features = model.backbone(images.tensors)
                if isinstance(features, torch.Tensor):
                    features = OrderedDict([("0", features)])
                features = list(features.values())
                anchors = model.anchor_generator(images, features)

                # show foreground anchors overlapped with ground truth and stochastic box (assuming one st. box.)
                # try underestimated st. box
                fig, ax = plt.subplots()
                ax.imshow(images.tensors[0].permute(1, 2, 0))
                for box, label in zip(targets[0]['boxes'], targets[0]['labels']):
                        [x1, y1, x2, y2] = box
                        box_h = (y2 - y1)
                        box_w = (x2 - x1)
                        bbox = patches.Rectangle((x1, y1), box_w, box_h,
                                                 linewidth=1, facecolor='none', edgecolor='g')
                        ax.add_patch(bbox)

                        # find center
                        center = [x1 + box_w/2, y1 + box_h/2]

                        # create stochastic box
                        ws = 0
                        hs = ws
                        x1 = center[0] - 0.5 * (model.bbox_priors['width_mean'][label] + ws *
                                                model.bbox_priors['width_std'][label])
                        x2 = center[0] + 0.5 * (model.bbox_priors['width_mean'][label] + ws *
                                                model.bbox_priors['width_std'][label])
                        y1 = center[1] - 0.5 * (model.bbox_priors['height_mean'][label] + hs *
                                                model.bbox_priors['height_std'][label])
                        y2 = center[1] + 0.5 * (model.bbox_priors['height_mean'][label] + hs *
                                                model.bbox_priors['height_std'][label])
                        st_box = torch.tensor([x1, y1, x2, y2], device=device)
                        bbox = patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1),
                                                 linewidth=1, facecolor='none', edgecolor='r')
                        ax.add_patch(bbox)


        return 1

        image_size = images.tensors.shape[-2:]
        grid_size = image_size[0] / features[-1].shape[-2:][0]  # square grid
        center = torch.tensor((image_size[1] / 2, image_size[0] / 2))

        xs = torch.linspace(-grid_size / 2, grid_size / 2, 5, device=device)
        ws, hs = torch.meshgrid(xs, xs, indexing="ij")
        x_centers = center[0] + ws
        y_centers = center[1] + hs
        centers = torch.stack([x_centers.flatten(), y_centers.flatten()], 1)

        img = Image.new("RGB", (image_size[1], image_size[0]))
        img1 = ImageDraw.Draw(img)
        for center in centers:
                img1.point([center[0], center[1]])
        img.show()

        # build stochastic boxes
        n = 2  # model.bbox_prior_coverage
        min_bbstep = 0.05
        max_bbstep = 1

        num_bbsteps = 5
        num_points = centers.shape[0]
        num_classes = data_loader.dataset.num_classes()
        fg_anchors = np.zeros((num_classes, num_bbsteps, num_points))
        for label in range(0, num_classes):
                for idx, bbp_steps in enumerate(torch.linspace(min_bbstep, max_bbstep, num_bbsteps, device=device)):
                        # start_time = time.time()
                        xs = torch.linspace(-n, n, int(torch.ceil(torch.tensor(2 * n / bbp_steps))) + 1, device=device)
                        ws, hs = torch.meshgrid(xs, xs, indexing="ij")
                        for idx2, center in enumerate(centers):
                                x1 = center[0] - 0.5 * (model.bbox_priors['width_mean'][label] + ws *
                                                        model.bbox_priors['width_std'][label])
                                x2 = center[0] + 0.5 * (model.bbox_priors['width_mean'][label] + ws *
                                                        model.bbox_priors['width_std'][label])
                                y1 = center[1] - 0.5 * (model.bbox_priors['height_mean'][label] + hs *
                                                        model.bbox_priors['height_std'][label])
                                y2 = center[1] + 0.5 * (model.bbox_priors['height_mean'][label] + hs *
                                                        model.bbox_priors['height_std'][label])

                                stochastic_boxes = torch.stack([x1.flatten(), y1.flatten(), x2.flatten(), y2.flatten()],
                                                               1)

                                outer_box = stochastic_boxes[-1, :].reshape(1, -1)
                                all_inner_matched_idx = model.proposal_matcher(utils.box_ioa(outer_box, anchors[0]))

                                all_inner_anchors = anchors[0][all_inner_matched_idx >= 0]
                                quality_matrix = box_ops.box_iou(stochastic_boxes, all_inner_anchors)

                                q2 = torch.max(quality_matrix, 0)[0]
                                qtemp = torch.zeros((1, len(anchors[0])), device=device)
                                qtemp[0, all_inner_matched_idx >= 0] = q2.reshape(1, -1)
                                matched_idx_2 = model.proposal_matcher(qtemp)
                                # matched_anchors = anchors[0][matched_idx_2 >= 0]
                                fg_anchors[label, idx, idx2] = sum(matched_idx_2 >= 0)

                                if 0:
                                        img = Image.new("RGB", (image_size[1], image_size[0]))
                                        img1 = ImageDraw.Draw(img)
                                        for anchor in matched_anchors:
                                                img1.rectangle(anchor.numpy(), outline="blue")
                                        img.show()

                        # total_time = time.time() - start_time
                        # total_time_str = str(datetime.timedelta(seconds=(total_time)))
                        # print(f"execution time {total_time_str}")

        for label in range(0, num_classes):
                plt.boxplot(fg_anchors[label].transpose(),
                            positions=torch.linspace(min_bbstep, max_bbstep, num_bbsteps))
        plt.show()

# print(torch.linspace(min_bbstep, max_bbstep, num_bbsteps))
# print(fg_anchors)
