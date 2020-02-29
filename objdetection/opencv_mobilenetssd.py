#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
implements object detection using OpenCV with a MobileNet SSD processing framework
'''

import os
import sys
import logging

import numpy as np
import cv2

MODEL_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
MODEL_COLORS = np.random.uniform(0, 255, size=(len(MODEL_CLASSES), 3))
DEFAULT_MODEL_PROTOTXT_FILE = "MobileNetSSD_deploy.prototxt.txt"
DEFAULT_MODEL_CAFFE_FILE = "MobileNetSSD_deploy.caffemodel"
CURDIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)

def load_additional_config_options(conf, cp):
    """load additional mobilenetssd specific config options out of our config file"""

    #validate notify_on_dataset_categories
    for e in conf['notify_on_dataset_categories']:
        if e not in MODEL_CLASSES:
            raise Exception("Invalid value '{}' is listed in supplied notify_on_dataset_categories config value. Valid options: {}".format(
                e, ', '.join(MODEL_CLASSES)))

def do_detections(conf, cp, img_attachment):
    load_additional_config_options(conf, cp)

    # load our serialized model from disk
    model_prototxt_file_path = os.path.join(CURDIR, DEFAULT_MODEL_PROTOTXT_FILE)
    assert os.path.exists(model_prototxt_file_path)
    model_caffe_file_path = os.path.join(CURDIR, DEFAULT_MODEL_CAFFE_FILE)
    assert os.path.exists(model_caffe_file_path)
    logger.info("Loading model {} (prototxt: {})...".format(DEFAULT_MODEL_CAFFE_FILE, DEFAULT_MODEL_PROTOTXT_FILE))
    net = cv2.dnn.readNetFromCaffe(model_prototxt_file_path, model_caffe_file_path)

    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    # (note: normalization is done via the authors of the MobileNet SSD
    # implementation)
    bytes_as_np_array = np.frombuffer(img_attachment.read(), dtype=np.uint8)
    image = cv2.imdecode(bytes_as_np_array, cv2.IMREAD_UNCHANGED)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and predictions
    logger.debug("Computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    found_objects = []
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        idx = int(detections[0, 0, i, 1])
        if MODEL_CLASSES[idx] != 'background':
            found_objects.append((MODEL_CLASSES[idx], confidence))

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence >= conf['min_confidence']:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(MODEL_CLASSES[idx], confidence * 100)
            cv2.rectangle(image, (startX, startY), (endX, endY),
                MODEL_COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, MODEL_COLORS[idx], 2)

    return found_objects, image
