#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
implements object detection using detectron2
originally taken from Detectron2 demo code at: https://github.com/facebookresearch/detectron2/blob/master/demo/

Licensed under the Apache License 2.0: https://github.com/facebookresearch/detectron2/blob/master/LICENSE
Original copyright notice: Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
'''

import os
import sys
import logging
import multiprocessing as mp
import os
import time
import tempfile
import pkg_resources
import atexit
import bisect
from collections import deque

import numpy as np
import cv2
import torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

DEFAULT_DETECTRON2_CONFIG_FILE = "configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml"
DEFAULT_DETECTRON2_USE_GPU = False
DEFAULT_DETECTRON2_EXTRA_OPTS = "MODEL.DEVICE cpu"

logger = logging.getLogger(__name__)

#make matplotlib logger not so chatty...
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.INFO)

class Visualization(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
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


#NOTE: THIS MAY NOT BE NECESSARY for single images
class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.

    <<UNMODIFIED FROM detectron2/demo/predictor.py>> 
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5


def load_additional_config_options(conf, cp):
    """load additional dectron2 specific config options out of our config file"""

    # the config file path should start with configs/
    conf['detectron2_config_file'] = cp.get('Default', 'detectron2_config_file', fallback=DEFAULT_DETECTRON2_CONFIG_FILE)
    if not conf['detectron2_config_file'].startswith("configs"):
        raise Exception("detectron2_config_file path should start with 'configs'")
    config_file_path = pkg_resources.resource_filename("detectron2", "model_zoo/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml")
    if not os.path.exists(config_file_path):
        raise Exception("Cannot find specified detectron2 config file at: {}".format(config_file_path))

    conf['detectron2_extra_opts'] = cp.get('Default', 'detectron2_extra_opts', fallback=DEFAULT_DETECTRON2_EXTRA_OPTS)
    conf['detectron2_extra_opts'] = conf['detectron2_extra_opts'].split()

    return config_file_path

def setup_cfg(conf, config_file_path):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file_path)
    cfg.merge_from_list(conf['detectron2_extra_opts'])
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = conf['min_confidence']
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf['min_confidence']
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = conf['min_confidence']
    cfg.freeze()
    return cfg

def do_detections(conf, cp, img_attachment):
    mp.set_start_method("spawn", force=True)
    config_file_path = load_additional_config_options(conf, cp)
    detectron2_cfg = setup_cfg(conf, config_file_path)

    # write img_attachment to file so that it works with detectron2's read_image method
    fp = tempfile.NamedTemporaryFile()
    fp.write(img_attachment.read())
    img = read_image(fp.name, format="BGR")
    fp.close()  # will remove the temp file

    # perform analysis
    visualization = Visualization(detectron2_cfg)
    # use PIL, to be consistent with evaluation
    start_time = time.time()
    predictions, visualized_output, metadata_classes = visualization.run_on_image(img)
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
