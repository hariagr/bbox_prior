import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import torch

def draw_boxes(image, target, predictions, data_loader, folder, score_thr = 0.5, topk = 100):
    # display
    image = image.permute(1, 2, 0).numpy()

    plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.imshow(image)

    # Find scored predicted boxes
    scores = predictions['scores']
    boxes = predictions['boxes']
    labels = predictions['labels']
    indices = np.where(scores >= score_thr)  # consider high scoring boxes for plots

    scores = scores[indices]
    boxes = boxes[indices]
    labels = labels[indices]

    # Find topk scored predicted boxes
    indices = np.argsort(-scores)
    indices = indices[:topk]

    scores = scores[indices]
    boxes = boxes[indices]
    labels = labels[indices]

    for box, label, score in zip(boxes, labels, scores):
        label = data_loader.dataset.label_to_name(label)
        [x1, y1, x2, y2] = box
        box_h = (y2 - y1)
        box_w = (x2 - x1)
        bbox = patches.Rectangle((x1, y1), box_w, box_h,
                                linewidth=2, facecolor='none', edgecolor='r')
        ax.add_patch(bbox)
        text = str(label) + ' ' + str(round(score, 2))
        ax.text(x1, y1, fontsize=10, s=text, color='0.2', verticalalignment='bottom',
                    bbox={'pad': 0, 'edgecolor': 'none', 'facecolor':'none'})

    # draw ground truth boxes
    for box, label in zip(target['boxes'], target['labels']):
        label = data_loader.dataset.label_to_name(label)
        [x1, y1, x2, y2] = box
        box_h = (y2 - y1)
        box_w = (x2 - x1)
        bbox = patches.Rectangle((x1, y1), box_w, box_h,
                                 linewidth=2, facecolor='none', edgecolor='g')
        ax.add_patch(bbox)
        ax.text(x1, y1, fontsize=8, s=str(label), color='0.2', verticalalignment='top',
                bbox={'pad': 0, 'edgecolor': 'none', 'facecolor': 'none'})

    os.makedirs(folder, exist_ok=True)
    filename = folder + '/pred_' + data_loader.dataset.image_index_to_image_file(target['image_id'][0])

    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close('all')

def draw_det(model, data_loader, device='cpu', folder=''):
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if 1: #torch.sum(torch.isin(targets[0]['labels'], )) > 0:
            detections = model(images)
            if 0:
                images, targets = model.transform(images, targets)
                # denormalize image
                dtype, device = images.tensors.dtype, images.tensors.device
                images = [images.tensors[k] for k in range(images.tensors.shape[0])]
                mean = torch.as_tensor(model.transform.image_mean, dtype=dtype, device=device)
                std = torch.as_tensor(model.transform.image_std, dtype=dtype, device=device)
                for k in range(len(images)):
                    images[k] = images[k] * std[:, None, None] + mean[:, None, None]
            detections = [{k: v.detach().numpy() for k, v in t.items()} for t in detections]
            targets = [{k: v.numpy() for k, v in t.items()} for t in targets]

            for image, target, detection in zip(images, targets, detections):
                draw_boxes(image, target, detection, data_loader, folder, score_thr=0.2)

