#-*- coding:utf-8 -*-
import torch
import random
from PIL import Image, ImageDraw
import torch.utils.data as data
import numpy as np
import random
import cv2
from utils.augmentations import preprocess

class DSNDataset(data.Dataset):
    """docstring for DSNDataset"""

    def __init__(self, list_file_s, list_file_t, mode='train'):
        super(DSNDataset, self).__init__()
        self.mode = mode
        self.fnames_s = []
        self.boxes_s = []
        self.labels_s = []
        self.fnames_t = []
        self.boxes_t = []
        self.labels_t = []

        self.fnames_s, self.boxes_s, self.labels_s = self.readFile(list_file_s)
        self.fnames_t, self.boxes_t, self.labels_t = self.readFile(list_file_t)
        self.num_target = len(self.fnames_t)
        self.num_samples = len(self.boxes_s)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img_s, face_target_s, head_target_s, \
            img_t, face_target_t, head_target_t = self.pull_item(index)
        return img_s, face_target_s, head_target_s,\
            img_t, face_target_t, head_target_t
    def pull_item(self, index):
        def getImgAndLabels(index, isSource):
            while True:
                if isSource:
                    image_path = self.fnames_s[index]
                    boxes = self.boxes_s[index]
                    labels = self.labels_s[index]
                else:
                    image_path = self.fnames_t[index]
                    boxes = self.boxes_t[index]
                    labels = self.labels_t[index]
                img = Image.open(image_path) #XXX I/O redundants?
                if img.mode == 'L':
                    img = img.convert('RGB')

                im_width, im_height = img.size
                boxes_s = self.annotransform(
                    np.array(boxes), im_width, im_height)
                label = np.array(labels)
                bbox_labels_s = np.hstack((label[:, np.newaxis], boxes_s)).tolist()
                img, sample_labels_s = preprocess(
                    img, bbox_labels_s, self.mode, image_path)
                sample_labels_s = np.array(sample_labels_s)
                if len(sample_labels_s) > 0:
                    face_target = np.hstack(
                        (sample_labels_s[:, 1:], sample_labels_s[:, 0][:, np.newaxis]))

                    assert (face_target[:, 2] > face_target[:, 0]).any()
                    assert (face_target[:, 3] > face_target[:, 1]).any()

                    #img = img.astype(np.float32)
                    face_box = face_target[:, :-1]
                    head_box = self.expand_bboxes(face_box)
                    head_target = np.hstack((head_box, face_target[
                                            :, -1][:, np.newaxis]))
                    break
                else:
                    if isSource:
                        index = random.randrange(0, self.num_samples)
                    else:
                        index = random.randrange(0, self.num_target)
            return img, face_target, head_target
        img_s, face_target_s, head_target_s = getImgAndLabels(index, True)
        index_t = random.randrange(0, self.num_target)
        img_t, face_target_t, head_target_t = getImgAndLabels(index_t, False)
        # print('>>> In dataset:')
        # print(img_s.shape, face_target_s.shape, head_target_s.shape)
        # print(img_t.shape, face_target_t.shape, head_target_t.shape)
        # img = Image.fromarray(img)
        '''
        draw = ImageDraw.Draw(img)
        w,h = img.size
        for bbox in sample_labels_s:
            bbox = (bbox[1:] * np.array([w, h, w, h])).tolist()

            draw.rectangle(bbox,outline='red')
        img.save('image.jpg')
        '''
        # XXX: For small GPU
        img_s = img_s.swapaxes(0,1).swapaxes(1,2)
        img_t = img_t.swapaxes(0,1).swapaxes(1,2)
        # print(img_s.shape)
        img_s = cv2.resize(img_s, (0,0), fx=0.05, fy=0.05)
        img_s = img_s.swapaxes(1,2).swapaxes(0,1)
        img_t = cv2.resize(img_t, (0,0), fx=0.05, fy=0.05)
        img_t = img_t.swapaxes(1,2).swapaxes(0,1)
        # img_s = Image.fromarray(img_s)
        # img_s = img_s.resize((320,320))
        # img_s = np.array(img_s)
        return torch.from_numpy(img_s), face_target_s, head_target_s, \
            torch.from_numpy(img_t), face_target_t, head_target_t

    def readFile(self, list_file):
        fnames = []
        boxes = []
        labels = []
        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split()
            num_faces = int(line[1])
            box = []
            label = []
            for i in range(num_faces):
                x = float(line[2 + 5 * i])
                y = float(line[3 + 5 * i])
                w = float(line[4 + 5 * i])
                h = float(line[5 + 5 * i])
                c = int(line[6 + 5 * i])
                if w <= 0 or h <= 0:
                    continue
                box.append([x, y, x + w, y + h])
                label.append(c)
            if len(box) > 0:
                fnames.append(line[0])
                boxes.append(box)
                labels.append(label)
        return fnames, boxes, labels

    def annotransform(self, boxes, im_width, im_height):
        boxes[:, 0] /= im_width
        boxes[:, 1] /= im_height
        boxes[:, 2] /= im_width
        boxes[:, 3] /= im_height
        return boxes

    def expand_bboxes(self,
                      bboxes,
                      expand_left=2.,
                      expand_up=2.,
                      expand_right=2.,
                      expand_down=2.):
        expand_bboxes = []
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            w = xmax - xmin
            h = ymax - ymin
            ex_xmin = max(xmin - w / expand_left, 0.)
            ex_ymin = max(ymin - h / expand_up, 0.)
            ex_xmax = max(xmax + w / expand_right, 0.)
            ex_ymax = max(ymax + h / expand_down, 0.)
            expand_bboxes.append([ex_xmin, ex_ymin, ex_xmax, ex_ymax])
        expand_bboxes = np.array(expand_bboxes)
        return expand_bboxes

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    face_targets_s = []
    head_targets_s = []
    face_targets_t = []
    head_targets_t = []

    imgs_s = []
    imgs_t = []
    for sample in batch:
        imgs_s.append(sample[0])
        face_targets_s.append(torch.FloatTensor(sample[1]))
        head_targets_s.append(torch.FloatTensor(sample[2]))
        imgs_t.append(sample[3])
        face_targets_t.append(torch.FloatTensor(sample[4]))
        head_targets_t.append(torch.FloatTensor(sample[5]))
    return torch.stack(imgs_s, 0), face_targets_s, head_targets_s,\
        torch.stack(imgs_t, 0), face_targets_t, head_targets_t



if __name__ == '__main__':
    from config import cfg
    dataset = DSNDataset(cfg.FACE.TRAIN_FILE)
    #for i in range(len(dataset)):
    dataset.pull_item(14)
