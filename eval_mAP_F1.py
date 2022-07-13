import numpy as np
import torch
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd
import time


def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    # print('a:',a)
    # print('b:',b)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    # area = area.cpu().numpy()

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = area + np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(dataset, retinanet, score_threshold=0.05, max_detections=900, save_path=None):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(dataset.num_classes())] for j in range(len(dataset))]

    retinanet.eval()

    with torch.no_grad():

        eval_time = np.zeros((len(dataset), 1))
        for index in range(len(dataset)):
            data = dataset[index]
            # print(data[0].size()
            # print(data[1])
            scale = 1.0

            # run network
            t = time.time()
            if torch.cuda.is_available():
                detect = retinanet(data[0].cuda().float().unsqueeze(dim=0))
            else:
                detect = retinanet([data[0]])
            eval_time[index] = time.time() - t

            scores = detect[0]['scores'].cpu().numpy()
            boxes = detect[0]['boxes'].cpu().numpy()
            labels = detect[0]['labels'].cpu().numpy()

            # scores = scores
            # labels = labels
            # boxes = boxes

            # correct boxes for image scale
            boxes /= scale

            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes = boxes[indices[scores_sort], :]
                image_scores = scores[scores_sort]
                image_labels = labels[indices[scores_sort]]
                image_detections = np.concatenate(
                    [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = np.zeros((0, 5))

            print('{}/{}'.format(index + 1, len(dataset)), end='\r')

    return all_detections, np.mean(eval_time)


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes() + 1)] for j in range(len(generator))]

    device = torch.device("cpu")
    res_list = []

    for idx in range(len(generator)):  #

        print('{:d}/{:d}'.format(idx + 1, len(generator)), end='\r')

        img, gt_target = generator[idx]

        for item in gt_target:
            gt_target[item] = gt_target[item].to(device)

        bxs = gt_target["boxes"]
        lbls = gt_target["labels"]
        mbxs = gt_target["mboxes"]
        mlbls = gt_target["mlabels"]

        if len(lbls) > 0:
            lbl = torch.reshape(lbls, (len(lbls), 1))
            annotations = torch.cat((bxs, lbl), 1)
            # copy detections to all_annotations
            for label in range(generator.num_classes()):
                all_annotations[idx][label] = annotations[annotations[:, 4] == label, :4]
        else:
            print('{:d},{:s}'.format(idx + 1, generator.image_names[idx]))
            raise NameError('number of GT objects are zero')

        if len(mlbls) > 0:
            mlbl = torch.reshape(mlbls, (len(mlbls), 1))
            annotations = torch.cat((mbxs, mlbl), 1)
            all_annotations[idx][generator.num_classes()] = annotations[annotations[:, 4] == generator.num_classes(),
                                                            :4]

        print('{:d}, {:d}, {:d}'.format(idx + 1, len(lbls), len(mlbls)), end='\r')

    return all_annotations


def evaluate(
        generator,
        retinanet,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=900,
        save_path=None,
        csv_path='eval_metrics',
        text_file_path=None,
        count=1,
        missedLabels=False,
        ret=1
):
    """ Evaluate a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet       : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    retinanet.eval()
    # gather all detections and annotations
    all_detections, eval_time = _get_detections(generator, retinanet, score_threshold=score_threshold,
                                                max_detections=max_detections, save_path=save_path)
    all_annotations = _get_annotations(generator)
    # print('detected: ',all_detections)
    # print('annotated: ',all_annotations)

    average_precisions = {}
    f1_score = {}
    tp = {}
    fp = {}
    fn = {}
    pr = {}
    rc = {}

    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0
        num_detections = 0.0
        num_ml_detections = 0.0
        num_ml_annotations = 0.0

        for i in range(len(generator)):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            num_detections += detections.shape[0]

            detected_annotations = []

            # missed label annotations
            ml_annotations = all_annotations[i][generator.num_classes()]
            ml_detections = []
            ml_detected_annotations = []
            if ml_annotations is not None:
                num_ml_annotations += ml_annotations.shape[0]

                for idx, d in enumerate(detections):

                    if ml_annotations.shape[0] == 0:
                        continue

                    overlaps = compute_overlap(np.expand_dims(d, axis=0), ml_annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in ml_detected_annotations:
                        ml_detections.append(idx)
                        ml_detected_annotations.append(assigned_annotation)

            detections_wo_ml = detections  # detections without those that are overlapping with missedlabel boxes
            if missedLabels and ml_annotations is not None:
                # print('{:d} detections overlapped with {:d} objects falls under missed class'.format(len(ml_detections), len(ml_annotations)))
                detections_wo_ml = np.delete(detections_wo_ml, ml_detections, axis=0)
                if ml_detections is not None:
                    num_ml_detections += len(ml_detections)

            # detections that matters for evaluations
            for d in detections_wo_ml:
                scores = np.append(scores, d[4])

                # if image does not have any object of the 'label' class,
                # all detections are false positive
                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            f1_score[label] = 0
            # tp[label] = 0
            # fp[label] = 0
            # fn[label] = 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]
        scores = scores[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        F1 = 2 / (1 / np.maximum(precision, np.finfo(np.float64).eps) + 1 / np.maximum(recall,
                                                                                       np.finfo(np.float64).eps))
        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

        print('\n###########################################################')
        print('\n{} - Ground Truths: {}'.format(generator.labels[label], num_annotations))
        print('{} - Detections: {} = TP: {} + FP: {} + ML: {}\n'.format(generator.label_to_name(label), num_detections,
                                                                        true_positives[-1], false_positives[-1],
                                                                        num_ml_detections))

        if text_file_path is not None:
            with open(text_file_path, "a") as text_file:
                text_file.write("\nEpochs {}:\n".format(count))
                text_file.write('{} - Ground Truths: {}\n'.format(generator.labels[label], num_annotations))
                text_file.write('{} - Detections: {} = TP: {} + FP: {} + ML: {}\n'.format(generator.label_to_name(label),
                                                                                          num_detections,
                                                                                          true_positives[-1],
                                                                                          false_positives[-1],
                                                                                          num_ml_detections))

        data = np.column_stack(
            (scores, true_positives, false_positives, num_annotations - true_positives, precision, recall, F1))

        columns = ['score', 'TP', 'FP', 'FN', 'Prec', 'Recall', 'F1']
        df = pd.DataFrame(data, columns=columns)

        sc = np.linspace(1, 0, 11)
        indx = []
        scc = []
        for k, ss in enumerate(sc[1:11]):
            try:
                indx.append(np.where(scores >= ss)[0][-1])
                scc.append(ss)
            except IndexError:
                print('no entry for score threshold >= {0:4.2}!\n'.format(ss))
        indx = np.array(indx)
        scc = np.array(scc)

        df1 = df.iloc[indx.reshape((-1,))]
        df2 = df1.copy()
        df2.loc[:, 'score'] = scc

        print(df2.reset_index(drop=True).to_string(
            formatters={'Prec': '{:8.2f}'.format, 'Recall': '{:8.2f}'.format, 'F1': '{:8.2f}'.format}))
        if text_file_path is not None:
            with open(text_file_path, "a") as text_file:
                text_file.write(" \n")
                text_file.write(df2.reset_index(drop=True).to_string(
                    formatters={'Prec': '{:8.2f}'.format, 'Recall': '{:8.2f}'.format, 'F1': '{:8.2f}'.format}))

        if ret:
            df2.set_index('score', inplace=True)
            kk = 0
            for index in df2.index:
                if index == 0.6:
                    kk += 1
                    f1_score[label] = df2.loc[index]['F1']
                    tp[label] = df2.loc[index]['TP']
                    fp[label] = df2.loc[index]['FP']
                    fn[label] = df2.loc[index]['FN']
                    pr[label] = df2.loc[index]['Prec']
                    rc[label] = df2.loc[index]['Recall']
            if kk == 0:  # when there are no entries for threshold>=0.6
                f1_score[label] = 0
                tp[label] = 0
                fp[label] = 0
                fn[label] = 0
                pr[label] = 0
                rc[label] = 0

    print('\nmAP:')
    if text_file_path is not None:
        with open(text_file_path, "a") as text_file:
            text_file.write("\nmAP:\n")
            for label in range(generator.num_classes()):
                label_name = generator.label_to_name(label)
                print('{0:5s}: {1:6.3f}'.format(label_name, average_precisions[label][0]))
                text_file.write('{0:5s}: {1:6.3f}\n'.format(label_name, average_precisions[label][0]))
    else:
        for label in range(generator.num_classes()):
            label_name = generator.label_to_name(label)
            print('{0:5s}: {1:6.3f}'.format(label_name, average_precisions[label][0]))

    # create a dataframe combining all return results
    # first create a dictionary
    #classes = ['pus', 'rbc', 'ep']
    results = {}
    for label in range(generator.num_classes()):
        class_name = generator.labels[label]
        results['mAP-' + class_name] = average_precisions[label][0]
        results['F1-' + class_name] = f1_score[label]
        results['TP-' + class_name] = tp[label]
        results['FP-' + class_name] = fp[label]
        results['FN-' + class_name] = fn[label]
        results['Pr-' + class_name] = pr[label]
        results['Rc-' + class_name] = rc[label]

    results = pd.DataFrame([results])

    retinanet.train()
    return results, eval_time
