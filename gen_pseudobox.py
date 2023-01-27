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

    c = []
    w = []
    h = []
    s = []
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

                    # for error analysis
                    [x1t, y1t, x2t, y2t] = gt_box.numpy()
                    err_c = np.sqrt(((x1 + 0.5*(x2 - x1)) - (x1t + 0.5*(x2t - x1t)))**2 + ((y1 + 0.5*(y2 - y1)) - (y1t + 0.5*(y2t - y1t)))**2)
                    c.append(err_c)
                    w.append(np.abs(x2 - x1 - (x2t - x1t)))
                    h.append(np.abs(y2 - y1 - (y2t - y1t)))
                    s.append(detection['scores'][torch.argmax(iou)])

    ndf = pd.read_csv(append_file)
    df = pd.concat([df, ndf])
    df.to_csv(folder, index=False)

    if 0:
        ndf = pd.DataFrame(columns=['s', 'c', 'w', 'h'])
        ndf['s'] = s
        ndf['c'] = c
        ndf['w'] = w
        ndf['h'] = h
        median = ndf.groupby(pd.cut(s, np.arange(0, 1, 0.1))).median()
        mad = ndf.groupby(pd.cut(s, np.arange(0, 1, 0.1))).mad()

        os.rmdir(folder)
        median.to_csv(folder + '_median.csv')
        mad.to_csv(folder + '_mad.csv')
