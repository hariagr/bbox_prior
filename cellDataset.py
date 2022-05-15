import torch
from six import raise_from
from torch.utils.data import Dataset
import sys
from PIL import Image
import numpy as np
import csv
import warnings
import pandas as pd


class CSVDataset(Dataset):
    """CSV dataset."""
    def __init__(self, train_file,
                 points_file=None,
                 class_list='data/annotations/classmaps.csv',
                 gclass_list='data/annotations/groupclassmaps.csv',
                 missedlabels=True,
                 transform=None,
                 weights=False):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.points_file = points_file
        self.class_list = class_list
        self.gclass_list = gclass_list
        self.transform = transform
        self.image_folder = 'data/images/'
        self.missedlabels = missedlabels
        self.mclass = 'missedlabel'

        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))  # dict: classes[pus]=0
        except ValueError as e:
            raise (ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)))

        self.labels = {}
        for key, value in self.classes.items():  # key=pus, value=0
            self.labels[value] = key  # labels[0]=pus, labels[1]=rbc, labels[2]=ep

        # parse the provided group class file
        try:
            self.glabels = {}
            self.gclasses = {}
            with self._open_for_csv(self.gclass_list) as file:
                self.gclasses = self.load_classes(csv.reader(file, delimiter=','))  # dict: classes[pus]=0
            for key, value in self.gclasses.items():  # key=pus, value=0
                self.glabels[value] = key  # labels[0]=c-pus, labels[1]=c-rbc, labels[2]=c-ep
        except:
            warnings.warn('invalid Group CSV class file. Running model without group annotations!!')

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                if csv.Sniffer().has_header(file.read(1024)):  # if header is present
                    file.seek(0)
                    next(file)  # skip header
                self.box_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes, self.gclasses, self.mclass)
        except ValueError as e:
            raise (ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)))
        self.image_names = list(self.box_data.keys())

        # # calculate number of cells per class
        train_data = pd.read_csv(self.train_file)
        train_cell_count = train_data.groupby('label')['image'].count()

        print("counts of ground truth boxes:")
        num_of_boxes = np.empty(self.num_classes(), dtype=int)
        for key, value in self.classes.items():
            num_of_boxes[value] = train_cell_count[key]
            print("%s: %d" % (key, train_cell_count[key]))

        # csv with img_path, x1, y1, x2, y2, class_name for point annotations
        if self.points_file is not None:
            try:
                with self._open_for_csv(self.points_file) as file:
                    if csv.Sniffer().has_header(file.read(1024)):  # if header is present
                        file.seek(0)
                        next(file)  # skip header
                    self.points_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes, self.gclasses,
                                                             self.mclass)
            except ValueError as e:
                raise (ValueError('invalid CSV annotations file: {}: {}'.format(self.points_file, e)))

            # include images that are annotated only with points
            for key in self.points_data.keys():
                if key not in self.image_names:
                    self.image_names.append(key)

            # # calculate number of cells per class
            points_data = pd.read_csv(self.points_file)
            points_cell_count = points_data.groupby('label')['image'].count()

            print("counts of point annotations:")
            num_of_points = np.empty(self.num_classes(), dtype=int)
            for key, value in self.classes.items():
                num_of_points[value] = points_cell_count[key]
                print("%s: %d" % (key, points_cell_count[key]))

            # self.stoc_boxes_per_class = torch.tensor([num_pus2, num_rbc2, num_ep2])
        else:
             num_of_points = np.empty(self.num_classes(), dtype=int)

        self.objects_per_class = torch.tensor(num_of_points + num_of_boxes)  # includes box and point annotated cells
        self.det_boxes_per_class = torch.tensor(num_of_boxes)
        self.stoc_boxes_per_class = torch.tensor(num_of_points)

        print('database initialization done...')

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def load_classes(self, csv_reader):
        result = {}  # dict: result[pus]=0

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise (ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))  # int(class_id)

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id  # result[pus]=0
        return result

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        boxes, labels, points, plabels, gboxes, glabels, mboxes, mlabels, \
                            weights, pweights, weights_imcls = self.load_annotations(idx)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = torch.zeros_like(labels)

        target["boxes"] = boxes
        target["labels"] = labels
        target["points"] = points
        target["plabels"] = plabels
        target["gboxes"] = gboxes
        target["glabels"] = glabels
        target["mboxes"] = mboxes
        target["mlabels"] = mlabels

        target["objects_per_class"] = self.objects_per_class
        target["det_boxes_per_class"] = self.det_boxes_per_class
        target["stoc_boxes_per_class"] = self.stoc_boxes_per_class
        target["image_name"] = torch.tensor([int(self.image_index_to_image_file(idx).split(".jpg")[0])])
        image, target = self.transform(img, target)

        return image, target

    def load_image(self, image_index):
        # print(self.image_names[image_index])
        img = Image.open(self.image_folder + self.image_names[image_index])
        return img

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list, points_list = None, None
        if self.image_names[image_index] in self.box_data.keys():
            annotation_list = self.box_data[self.image_names[image_index]]
        if self.points_file is not None:
            if self.image_names[image_index] in self.points_data.keys():
                points_list = self.points_data[self.image_names[image_index]]

        boxes = []
        labels = []
        points = []
        plabels = []
        gboxes = []
        glabels = []
        mboxes = []
        mlabels = []
        weights = []
        pweights = []
        weights_imcls = []

        # some images appear to miss annotations
        if annotation_list is not None and len(annotation_list) == 0 and \
                points_list is not None and len(points_list) == 0:
            return boxes, labels, points, plabels, gboxes, glabels, mboxes, mlabels, weights, pweights, weights_imcls

        # parse annotations
        if annotation_list is not None:  # box annotations
            for idx, a in enumerate(annotation_list):
                # some annotations have basically no width / height, skip them
                x1 = a['x1']
                x2 = a['x2']
                y1 = a['y1']
                y2 = a['y2']

                if (x2 - x1) < 1 or (y2 - y1) < 1:
                    continue

                if a['class'] in self.classes:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(self.classes[a['class']])
                    weights.append(a['weight'])
                    # weights_imcls.append(self.weights_imcls[a['class']])
                elif a['class'] in self.mclass:
                    if self.missedlabels:
                        mboxes.append([x1, y1, x2, y2])
                        mlabels.append(len(self.classes))  # not of use but included for later use if need comes up
                elif a['class'] in self.gclasses:
                    gboxes.append([x1, y1, x2, y2])
                    glabels.append(self.gclasses[a['class']])

        if points_list is not None:
            for idx, a in enumerate(points_list):  # point annotations
                if a['class'] in self.classes:
                    x1 = a['x1']
                    y1 = a['y1']
                    #x2 = a['x1']
                    #y2 = a['y1']
                    points.append([x1, y1])  # store only center of the box
                    plabels.append(self.classes[a['class']])
                    pweights.append(a['weight'])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        points = torch.as_tensor(points, dtype=torch.float32)
        plabels = torch.as_tensor(plabels, dtype=torch.int64)

        gboxes = torch.as_tensor(gboxes, dtype=torch.float32)
        glabels = torch.as_tensor(glabels, dtype=torch.int64)

        mboxes = torch.as_tensor(mboxes, dtype=torch.float32)
        mlabels = torch.as_tensor(mlabels, dtype=torch.int64)

        weights = torch.as_tensor(weights, dtype=torch.float32)
        pweights = torch.as_tensor(pweights, dtype=torch.float32)
        weights_imcls = torch.as_tensor(weights_imcls, dtype=torch.float32)

        return boxes, labels, points, plabels, gboxes, glabels, mboxes, mlabels, weights, pweights, weights_imcls

    def _read_annotations(self, csv_reader, classes, gclasses, mclass):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                # if self.weights:
                #     img_file, x1, y1, x2, y2, class_name, weight = row[:7]
                # else:
                img_file, x1, y1, x2, y2, class_name = row[:6]
                weight = 1
            except ValueError:
                raise_from(ValueError(
                    'line {}: format should be \'img_file,x1,y1,x2,y2,class_name,weights\' or \'img_file,,,,,\''.format(line)),
                           None)

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path with no box coordinates, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = self._parse(int(float(x1)), int, 'line {}: malformed x1: {{}}'.format(line))  # int(x1) as x1 is in str
            y1 = self._parse(int(float(y1)), int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(int(float(x2)), int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(int(float(y2)), int, 'line {}: malformed y2: {{}}'.format(line))

            weight = self._parse((float(weight)), float, 'line {}: malformed weight: {{}}'.format(line))  # float(weight)

            # Check that the bounding box is valid.
            # if x2 <= x1:
            #     raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            # if y2 <= y1:
            #     raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            # check if the current class name is correctly present
            #if (class_name not in classes) and (class_name not in gclasses):
            #    raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

            if (class_name in classes) or (class_name in gclasses) or (class_name in mclass):
                result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name, 'weight': weight})

        return result  # list of images having a dictionary for every bbox

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def glabel_to_name(self, label):
        return self.glabels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)

    def image_index_to_image_file(self, image_index):
        return self.image_names[image_index]
        
    def compute_priors(self):
        print(len(self.box_data))
        pus_data = []
        rbc_data = []
        ep_data = []
        for id in range(len(self.box_data)):
            annotation_list = self.box_data[self.image_names[id]]
            for idx, a in enumerate(annotation_list):
                wd = a['x2'] - a['x1']
                ht = a['y2'] - a['y1']
                if a['class'] == 'pus':
                    pus_data.append([wd, ht])
                elif a['class'] == 'rbc':
                    rbc_data.append([wd, ht])
                elif a['class'] == 'ep':
                    ep_data.append([wd, ht])

        self.priors[0].append(np.mean(pus_data, 0))  # pus
        self.priors[0].append(np.std(pus_data, 0))
        self.priors[1].append(np.mean(rbc_data, 0))  # rbc
        self.priors[1].append(np.std(rbc_data, 0))
        self.priors[2].append(np.mean(ep_data, 0))  # ep
        self.priors[2].append(np.std(ep_data, 0))
        print('priors ', self.priors)

        return self.priors


    def compute_log_priors(self):

        pus_data = []
        rbc_data = []
        ep_data = []
        for id in range(len(self.box_data)):
            annotation_list = self.box_data[self.image_names[id]]
            for idx, a in enumerate(annotation_list):
                wd = np.log(a['x2'] - a['x1'])
                ht = np.log(a['y2'] - a['y1'])
                if a['class'] == 'pus':
                    pus_data.append([wd, ht])
                elif a['class'] == 'rbc':
                    rbc_data.append([wd, ht])
                elif a['class'] == 'ep':
                    ep_data.append([wd, ht])

        self.log_priors[0].append(np.mean(pus_data, 0))  # pus
        self.log_priors[0].append(np.std(pus_data, 0))
        self.log_priors[1].append(np.mean(rbc_data, 0))  # rbc
        self.log_priors[1].append(np.std(rbc_data, 0))
        self.log_priors[2].append(np.mean(ep_data, 0))  # ep
        self.log_priors[2].append(np.std(ep_data, 0))
        print('Log priors ', self.log_priors)

        return self.log_priors