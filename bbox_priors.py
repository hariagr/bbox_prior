import torch
import pandas as pd


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

    model.bbox_priors = bbox_priors
    model.head.bbox_priors = bbox_priors
    model.head.regression_head.bbox_priors = bbox_priors

    print(bbox_priors)

    return model
