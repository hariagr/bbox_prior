import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import torch
import pandas as pd
from torchvision.ops import boxes as box_ops
import os

def gen_box_from_point(model, data_loader, device='cpu', folder='', append_file=''):
    df = pd.DataFrame(columns=['image', 'xmin', 'ymin', 'xmax', 'ymax', 'label'])
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        detections = model(images)
        detections = [{k: v.detach() for k, v in t.items()} for t in detections]
        #targets = [{k: v.numpy() for k, v in t.items()} for t in targets]

        for image, target, detection in zip(images, targets, detections):
            image_name = data_loader.dataset.image_index_to_image_file(target['image_id'][0])
            for (gt_box, label) in zip(target['boxes'], target['labels'].numpy()):
                iou = box_ops.box_iou(gt_box.reshape(1, -1), detection['boxes'])
                if iou.numel() > 0 and torch.max(iou) >= 0.5:
                    pseudo_box = detection['boxes'][torch.argmax(iou)]
                    [x1, y1, x2, y2] = pseudo_box.numpy()
                    df.loc[len(df.index)] = [image_name, x1, y1, x2, y2, data_loader.dataset.labels[label]]
                # top_score = 0.05
                # for det_box, score in zip(detection['boxes'], detection['scores']):
                #     [x1, y1, x2, y2] = box
                #     xc = 0.5 * (x1 + x2)
                #     yc = 0.5 * (y1 + y2)
                #     if (point[0] > x1 and point[0] < x2) and (point[1] > y1 and point[1] < y2):
                #         if score > top_score:
                #             pseudo_box = box
                #             top_score = score

    ndf = pd.read_csv(append_file)
    df = pd.concat([df, ndf])

    os.rmdir(folder)
    df.to_csv(folder, index=False)
