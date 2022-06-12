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

                    tl = torch.cat((gt_widths, gt_heights, torch.log(gt_widths), torch.log(gt_heights), target["labels"].reshape(-1,1)), 1)
                    table = torch.cat((table, tl), 0)

        except Exception as e:
            print(e)
            continue

    # compute priors
    df = pd.DataFrame(table.cpu().numpy(), columns=["width", "height", "logOfwidth", "logOfheight", "labels"])
    gdf = df.groupby('labels')
    bbox_priors = {}
    bbox_priors["width_mean"] = torch.as_tensor(gdf['width'].mean().values, dtype=torch.float32)
    bbox_priors["width_std"] = torch.as_tensor(gdf['width'].std().values, dtype=torch.float32)
    bbox_priors["height_mean"] = torch.as_tensor(gdf['height'].mean().values, dtype=torch.float32)
    bbox_priors["height_std"] = torch.as_tensor(gdf['height'].std().values, dtype=torch.float32)

    bbox_priors["logOfwidth_mean"] = torch.as_tensor(gdf['logOfwidth'].mean().values, dtype=torch.float32)
    bbox_priors["logOfwidth_std"] = torch.as_tensor(gdf['logOfwidth'].std().values, dtype=torch.float32)
    bbox_priors["logOfheight_mean"] = torch.as_tensor(gdf['logOfheight'].mean().values, dtype=torch.float32)
    bbox_priors["logOfheight_std"] = torch.as_tensor(gdf['logOfheight'].std().values, dtype=torch.float32)

    model.bbox_priors = bbox_priors
    model.head.bbox_priors = bbox_priors
    model.head.regression_head.bbox_priors = bbox_priors

    print(bbox_priors)

    return model
