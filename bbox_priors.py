import numpy as np
from noisyopt import minimizeCompass
import pandas as pd
from scipy import stats
from torchvision.ops import boxes as box_ops
import torch

def cal_bbox_priors(model, dataloader, device):
    model.train()
    table = torch.tensor([], device=device)
    for images, targets in dataloader:
        try:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            images, targets = model.transform(images, targets)

            for target in targets:
                if target["hasBoxes"]:
                    reference_boxes = target["boxes"]
                    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
                    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
                    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
                    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)
                    gt_widths = reference_boxes_x2 - reference_boxes_x1
                    gt_heights = reference_boxes_y2 - reference_boxes_y1

                    tl = torch.cat((gt_widths, gt_heights, torch.log(gt_widths), torch.log(gt_heights),
                                    target["labels"].reshape(-1, 1)), 1)
                    table = torch.cat((table, tl), 0)

        except Exception as e:
            print(e)
            continue

    # compute priors
    df = pd.DataFrame(table.cpu().numpy(), columns=["width", "height", "logOfwidth", "logOfheight", "labels"])
    gdf = df.groupby('labels')

    num_classes = dataloader.dataset.num_classes()
    bbox_priors = {"width_mean": torch.zeros(num_classes, dtype=torch.float32),
                   "width_std": torch.zeros(num_classes, dtype=torch.float32),
                   "height_mean": torch.zeros(num_classes, dtype=torch.float32),
                   "height_std": torch.zeros(num_classes, dtype=torch.float32),
                   "logOfwidth_mean": torch.zeros(num_classes, dtype=torch.float32),
                   "logOfwidth_std": torch.zeros(num_classes, dtype=torch.float32),
                   "logOfheight_mean": torch.zeros(num_classes, dtype=torch.float32),
                   "logOfheight_std": torch.zeros(num_classes, dtype=torch.float32),
                   "width_mode": torch.zeros(num_classes, dtype=torch.float32),
                   "height_mode": torch.zeros(num_classes, dtype=torch.float32),
                   "width_mean_IOU": torch.zeros(num_classes, dtype=torch.float32),
                   "height_mean_IOU": torch.zeros(num_classes, dtype=torch.float32),
                   }

    for idx, df_grp in gdf:
        idx = int(idx)

        bbox_priors["width_mean"][idx] = torch.as_tensor(df_grp['width'].mean(), dtype=torch.float32)
        bbox_priors["width_std"][idx] = torch.as_tensor(df_grp['width'].std(), dtype=torch.float32)
        bbox_priors["height_mean"][idx] = torch.as_tensor(df_grp['height'].mean(), dtype=torch.float32)
        bbox_priors["height_std"][idx] = torch.as_tensor(df_grp['height'].std(), dtype=torch.float32)

        bbox_priors["logOfwidth_mean"][idx] = torch.as_tensor(df_grp['logOfwidth'].mean(), dtype=torch.float32)
        bbox_priors["logOfwidth_std"][idx] = torch.as_tensor(df_grp['logOfwidth'].std(), dtype=torch.float32)
        bbox_priors["logOfheight_mean"][idx] = torch.as_tensor(df_grp['logOfheight'].mean(), dtype=torch.float32)
        bbox_priors["logOfheight_std"][idx] = torch.as_tensor(df_grp['logOfheight'].std(), dtype=torch.float32)

        bbox_priors["width_mode"][idx] = torch.as_tensor(cal_mode(df_grp['width']), dtype=torch.float32)
        bbox_priors["height_mode"][idx] = torch.as_tensor(cal_mode(df_grp['height']), dtype=torch.float32)

        bbox = cal_mean_IOU_box(df_grp)
        bbox_priors["width_mean_IOU"][idx] = torch.as_tensor(bbox[0], dtype=torch.float32)
        bbox_priors["height_mean_IOU"][idx] = torch.as_tensor(bbox[1], dtype=torch.float32)

    model.bbox_priors = bbox_priors
    model.head.bbox_priors = bbox_priors
    model.head.regression_head.bbox_priors = bbox_priors

    print(bbox_priors)

    return model

def cal_mode(data, num_points = 500):
    data = data.to_numpy().reshape(1, -1)
    kernel = stats.gaussian_kde(data)
    X = np.linspace(np.min(data), np.max(data), num_points).reshape(1, -1)
    prob = kernel(X.reshape(1, -1))
    mode = X[0, np.argmax(prob)]
    return mode

    # plt.plot(X_plot, prob)
    # mean = np.mean(width)
    # mean2 = np.sum(prob*X_plot)/np.sum(prob)

def cal_mean_IOU_box(df):

    def cal_loss(x, bboxa, prob):
        bboxb = torch.tensor([-0.5 * x[0], -0.5 * x[1], 0.5 * x[0], 0.5 * x[1]]).reshape(-1, 1)
        iou = box_ops.box_iou(bboxa.T, bboxb.T)
        L = - (1 / bboxa.shape[1]) * np.sum(np.multiply(iou.numpy().flatten(), prob.flatten()))
        return L

    num_points = 100

    wh = np.stack([df['width'], df['height']])
    kernel = stats.gaussian_kde(wh)

    mu = np.array([stats.tmean(wh[0]), stats.tmean(wh[1])])
    std = np.array([stats.tstd(wh[0]), stats.tstd(wh[1])])

    ub = np.array((mu + 3 * std, wh.max(axis=1))).min(axis=0)
    lb = np.array((mu - 3 * std, wh.min(axis=1))).max(axis=0)

    x = np.linspace(lb[0], ub[0], num_points).reshape(1, -1)
    y = np.linspace(lb[1], ub[1], num_points).reshape(1, -1)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    xi = np.array([xv.flatten(), yv.flatten()])
    prob = kernel(xi)

    xi = torch.tensor(np.array([xv.flatten(), yv.flatten()]))
    bboxa = torch.stack([-0.5 * xi[0], -0.5 * xi[1], 0.5 * xi[0], 0.5 * xi[1]])

    obj = lambda x: cal_loss(x, bboxa, prob)
    bounds = np.array([[lb[0], ub[0]], [lb[1], ub[1]]])
    res = minimizeCompass(obj, x0=mu, bounds=bounds, deltatol=1e-6, paired=False, errorcontrol=False, disp=False)

    print(f"x0: {mu}, x*: {res.x}, d: {np.linalg.norm(mu-res.x)}")

    return res.x
