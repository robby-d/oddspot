# -*- coding: utf-8 -*-
'''
implements object detection using detectron2

Contains segements of code originally from Detectron2 demo code at:
    https://github.com/facebookresearch/detectron2/blob/master/demo/
Licensed under the Apache License 2.0: https://github.com/facebookresearch/detectron2/blob/master/LICENSE
Original copyright notice: Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
'''

import os
import sys
import logging
import os
import time
import tempfile
import pkg_resources

import numpy as np
import cv2
import torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

DEFAULT_DETECTRON2_CONFIG_FILE = "configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml"

logger = logging.getLogger(__name__)

#make matplotlib logger not so chatty...
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.INFO)

class Detector(object):
    def __init__(self, conf, cp, gpu_id=0, instance_mode=ColorMode.IMAGE):
        """gpu_id of 0 for CPU, otherwise 1, 2, 3, etc for the GPU in the system to use"""
        num_gpu = torch.cuda.device_count()
        if gpu_id > num_gpu:
            raise Exeption("Invalid GPU ID of {} (num_gpu = {})".format(gpu_id, num_gpu))
        self.gpu_id = gpu_id

        #load config
        self.config_file_path = self._load_additional_config_options(conf, cp)
        self.detectron2_cfg = self._setup_cfg(conf)
        self.metadata = MetadataCatalog.get(
            self.detectron2_cfg.DATASETS.TEST[0] if len(self.detectron2_cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.predictor = DefaultPredictor(self.detectron2_cfg)

    def _load_additional_config_options(self, conf, cp):
        """load additional dectron2 specific config options out of our config file"""

        # the config file path should start with configs/
        conf['detectron2_config_file'] = cp.get('detection', 'detectron2_config_file', fallback=DEFAULT_DETECTRON2_CONFIG_FILE)
        if not conf['detectron2_config_file'].startswith("configs"):
            raise Exception("detectron2_config_file path should start with 'configs'")
        config_file_path = pkg_resources.resource_filename("detectron2", "model_zoo/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml")
        if not os.path.exists(config_file_path):
            raise Exception("Cannot find specified detectron2 config file at: {}".format(config_file_path))

        conf['detectron2_extra_opts'] = cp.get('detection', 'detectron2_extra_opts', fallback='')
        conf['detectron2_extra_opts'] = conf['detectron2_extra_opts'].split()
        return config_file_path

    def _setup_cfg(self, conf):
        assert self.config_file_path
        # load config from file and command-line arguments
        cfg = get_cfg()
        cfg.merge_from_file(self.config_file_path)
        cfg.merge_from_list(conf['detectron2_extra_opts'])
        
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = conf['min_confidence']
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf['min_confidence']
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = conf['min_confidence']

        # only override MODEL.DEVICE if not explicitly specified via detectron2_extra_opts
        logger.debug("Initial MODEL.DEVICE setting is: {}".format(cfg.MODEL.DEVICE))
        if not cfg.MODEL.DEVICE or cfg.MODEL.DEVICE.lower() == 'cuda':
            # multi GPU support
            cfg.MODEL.DEVICE = "cuda:{}".format(self.gpu_id - 1) if self.gpu_id > 0 else "cpu"
            # CUDA device ordinal is zero based, whereas self.gpu_id is 1 based
        logger.debug("Modified MODEL.DEVICE setting is: {}".format(cfg.MODEL.DEVICE))

        cfg.freeze()
        return cfg

    def _run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output, self.metadata.thing_classes

    def do_detections(self, img_attachment):
        # write img_attachment to file so that it works with detectron2's read_image method
        fp = tempfile.NamedTemporaryFile()
        fp.write(img_attachment.read())
        img = read_image(fp.name, format="BGR")
        fp.close()  # will remove the temp file

        # perform analysis
        # use PIL, to be consistent with evaluation
        start_time = time.time()
        predictions, visualized_output, metadata_classes = self._run_on_image(img)
        logger.info("{} in {:.2f}s".format(
                "Detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions else "Detected zero instances",
                time.time() - start_time,
            )
        )

        #TODO: support panoptic and semantic segmentation results ("panoptic_seg"/"sem_seg" in predictions)
        found_objects = []
        for i in range(len(predictions["instances"])):
            found_objects.append((metadata_classes[predictions["instances"].pred_classes[i]], float(predictions["instances"].scores[i])))

        image = visualized_output.get_image()[:, :, ::-1]
        return found_objects, image
