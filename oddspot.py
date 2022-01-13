#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import datetime
import configparser
import argparse
import pickle
import logging
import logging.handlers
import email
import json
import sys
import requests
import traceback
import asyncio
import queue
import threading
import atexit
import signal
import socket
import random
from builtins import str

from memory_profiler import profile
import objgraph

import pushover
import torch
import cv2
import cmapy
import numpy as np
from aiosmtpd.controller import Controller as AiosmtpdController
from deepstack_sdk import ServerConfig, Detection

import util

# memory leak tracing
import tracemalloc

PROG_NAME = "oddspot"
CURDIR = os.path.dirname(os.path.realpath(__file__))
STATE_FILE = os.path.join(CURDIR, "{}.dat".format(PROG_NAME))
LOG_FILE = os.path.join(CURDIR, "logs", "{}.log".format(PROG_NAME))
CONF_FILE = os.path.join(CURDIR, "{}.ini".format(PROG_NAME))

DEFAULT_MIN_CONFIDENCE = 0.7  # 70%
YOLOV5_MODEL_CHOICES = ('yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x','yolov5n6', 'yolov5s6', 'yolov5m6', 'yolov5l6', 'yolov5x6')
DEFAULT_YOLOV5_MODEL = 'yolov5m'
DEFAULT_MIN_NOTIFY_PERIOD = 600  # in seconds (600 = 10 minutes)
PUSHOVER_MAX_ATTACHMENT_SIZE = 2621440  # 2.5MB
PLATE_RECOGNIZER_ALLOWED_DETECTION_CATEGORIES = ('car', 'truck', 'bus', 'motorcycle')
DEEPSTACK_DEFAULT_API_PORT = 5000

# globals
logger = logging.getLogger(__name__)
conf = None
state = None
cleanup_called = False


def load_config():
    if not os.path.exists(CONF_FILE):
        raise Exception("Config file does not exist at path: '{}'".format(CONF_FILE))
    conf = {}
    cp = configparser.ConfigParser()
    cp.read(CONF_FILE)

    # section: detection
    conf['min_confidence'] = cp.getfloat('detection', 'min_confidence', fallback=DEFAULT_MIN_CONFIDENCE)
    assert conf['min_confidence'] > 0.0 and conf['min_confidence'] <= 1.0

    #conf['yolov5_model'] = cp.get('detection', 'yolov5_model', fallback=DEFAULT_YOLOV5_MODEL)
    #if conf['yolov5_model'] not in YOLOV5_MODEL_CHOICES:
    #    raise Exception("Invalid YOLOv5 model, must be one of: {}".format(', '.join(YOLOV5_MODEL_CHOICES)))

    conf['deepstack_api_port'] = cp.getint('detection', 'deepstack_api_port', fallback=DEEPSTACK_DEFAULT_API_PORT)
    assert conf['deepstack_api_port'] < 65535

    # section: notify
    conf['pushover_user_key'] = cp.get('notify', 'pushover_user_key')
    conf['pushover_api_token'] = cp.get('notify', 'pushover_api_token')

    conf['min_notify_period'] = cp.getint('notify', 'min_notify_period', fallback=DEFAULT_MIN_NOTIFY_PERIOD)
    assert conf['min_notify_period'] >= 0

    conf['notify_on_dataset_categories'] = cp.get('notify', 'notify_on_dataset_categories')
    conf['notify_on_dataset_categories'] = [e.strip() for e in conf['notify_on_dataset_categories'].split(',')]

    # section: smtpd
    conf['smtp_listen_host'] = cp.get('smtpd', 'listen_host').strip()
    conf['smtp_listen_host'] = '' if not conf['smtp_listen_host'] else conf['smtp_listen_host']
    # ^ use '' to bind dual stack (ipv4 and v6) on all interfaces, by default
    conf['smtp_listen_port'] = cp.getint('smtpd', 'listen_port')

    # section: integrations
    conf['platerecognizer_api_key'] = cp.get('integrations', 'platerecognizer_api_key').strip()
    conf['platerecognizer_regions_hint'] = cp.get('integrations', 'platerecognizer_regions_hint')
    if conf['platerecognizer_regions_hint']:
        conf['platerecognizer_regions_hint'] = json.loads(conf['platerecognizer_regions_hint'])
    else:
        conf['platerecognizer_regions_hint'] = []
    assert isinstance(conf['platerecognizer_regions_hint'], (list, tuple))

    # section: cameras
    conf['camera_names_from_sender'] = cp.get('cameras', 'camera_names_from_sender', fallback={})
    if not conf['camera_names_from_sender']:
        raise Exception('camera_names_from_sender must be defined')
    conf['camera_names_from_sender'] = json.loads(conf['camera_names_from_sender'])
    assert isinstance(conf['camera_names_from_sender'], dict)

    conf['camera_custom_configs'] = cp.get('cameras', 'camera_custom_configs', fallback={})
    conf['camera_custom_configs'] = json.loads(conf['camera_custom_configs'])
    assert isinstance(conf['camera_custom_configs'], dict)

    # section: other
    conf['memory_usage_logging'] = cp.getboolean('other', 'memory_usage_logging', fallback=False)

    return conf, cp

def handle_exception(type, value, tb):
    if issubclass(type, KeyboardInterrupt):
        sys.__excepthook__(type, value, tb)
        return

    error_traceback_str = ''.join(traceback.format_exception(type, value, tb))

    # grab the locals for the stack frame with the exception
    error_locals = []
    while tb.tb_next:
        tb = tb.tb_next
    for k, v in tb.tb_frame.f_locals.items():
            try:
                error_locals.append("\t{:.40}: {:.255}\n".format(k, v))
            except:
                error_locals.append("\t{:.40}: CANNOT PRINT VALUE\n".format(k))
    error_locals_str = ''.join(error_locals)
    logger.critical("Uncaught exception", exc_info=(type, value, tb))
    logger.info(error_locals_str)

class SMTPDHandler:
    def __init__(self, processing_queue):
        self.processing_queue = processing_queue
        logging.getLogger('mail.log').setLevel(logging.WARN)

    async def handle_RCPT(self, server, session, envelope, address, rcpt_options):
        allowed_addresses = ('oddspot@localhost', 'oddspot@localhost.localdomain', 'oddspot@{}'.format(socket.gethostname()), 'oddspot@{}'.format(socket.getfqdn()))
        if address.lower() not in allowed_addresses:
            logger.debug("not relaying message to {} (allowed recipients are: {})".format(address.lower(), ', '.join(allowed_addresses)))
            return '550 not relaying to that user and domain'
        envelope.rcpt_tos.append(address)
        return '250 OK'

    async def handle_DATA(self, server, session, envelope):
        #throw message info into a queue for processing
        self.processing_queue.put(envelope)
        return '250 Message accepted for delivery'

def email_worker_iter(thread_index, processing_queue, detector):
    global conf, state

    worker_logger = logging.getLogger()
    #worker_logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    #formatter = logging.Formatter('worker{}')
    #file_handler.setFormatter(formatter)

    logger.debug("worker{:02}: Waiting on mail message...".format(thread_index))
    logger.debug("state is: {}".format(state))
    # block until we have a new incoming message
    envelope = processing_queue.get()
    if envelope is None:  #main thread is telling us to stop
        logger.debug("worker{:02}: Thread received sign to terminate".format(thread_index))
        return False

    utc_now = datetime.datetime.utcnow()
    utc_now_epoch_timestamp = utc_now.timestamp()
    utc_now_epoch = int(utc_now_epoch_timestamp)  # second precision

    email_str = envelope.content.decode('utf8', errors='replace')
    parsed_email = util.email_parse(email_str)

    # parse the raw image data out of the email
    logger.info("worker{:02}: Email from: '{}', subject: '{}'".format(thread_index, parsed_email['from'], parsed_email['subject']))
    logger.info("worker{:02}: Email has {} attachments: {}".format(thread_index, len(parsed_email['attachments']),
        ', '.join(['{}: {} (size: {}) ({})'.format(i, a.content_type, util.convert_size(a.size), 'USING' if i==0 else 'NOT USING')
            for i, a in enumerate(parsed_email['attachments'])]) if len(parsed_email['attachments']) else 'N/A'))
    if not parsed_email['attachments'] or parsed_email['attachments'][0].content_type not in ('application/octet-stream', 'image/jpeg', 'image/png'):
        logger.warn("worker{:02}: Cannot parse out image from email".format(thread_index))
        return True
    # use first attachment
    img_attachment_raw = parsed_email['attachments'][0]
    img_attachment_raw.seek(0)  # just in case
    bytes_as_np_array = np.frombuffer(img_attachment_raw.read(), dtype=np.uint8)
    image = cv2.imdecode(bytes_as_np_array, cv2.IMREAD_UNCHANGED)    

    # determine which camera sent us this image (using the entire from email address if no friendly name in 'camera_names_from_sender')
    camera_name = conf['camera_names_from_sender'].get(parsed_email['from'], parsed_email['from'])
    state['last_notify'].setdefault(camera_name, False)
    logger.info("worker{:02}: Identified camera as {}".format(thread_index, camera_name))

    # see if we should even run image analysis
    assert state['last_notify'][camera_name] is False or utc_now_epoch - state['last_notify'][camera_name] >= 0
    if state['last_notify'][camera_name] is not False and utc_now_epoch - state['last_notify'][camera_name] < conf['min_notify_period']:
        logger.info("worker{:02}: Skipping image analysis as last notification for this camera is {} seconds ago (needs to be >= {} seconds)".format(
            thread_index, utc_now_epoch - state['last_notify'][camera_name], conf['min_notify_period']))
        return True

    # see if we should always forward the image along...
    always_notify = False
    if camera_name in conf['camera_custom_configs'] and conf['camera_custom_configs'][camera_name].get('always_notify', False):
        logger.info("worker{:02}: Notifying on image irregardless of analysis (always_notify option enabled for this camera '{}')".format(
            thread_index, camera_name))
        always_notify = True

    #perform detection
    detection_start = time.time()
    response = detector.detectObject(image, min_confidence=conf['min_confidence'])
    detection_end = time.time()
    found_objects = [(obj.label, obj.confidence) for obj in response]
    #mark up image
    for obj in response:
        #properties: obj.label, obj.confidence, obj.x_min, obj.y_min, obj.x_max, obj.y_max
        color_fg = cmapy.color('plasma', random.randrange(0, 256), rgb_order=False)

        # draw rectangle around recognized object
        cv2.rectangle(image, (obj.x_min, obj.y_min), (obj.x_max, obj.y_max), color_fg, 2, cv2.LINE_AA)

        # add label to rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0  # or 0.5
        font_thickness = 2
        text_color_bg=(0, 0, 0)  # black
        label = "{}: {:.2f}%".format(obj.label, obj.confidence * 100)
        text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_w, text_h = text_size

        y = obj.y_min - text_h if obj.y_min - text_h > text_h else obj.y_min + text_h
        # draw background rectangle for text
        cv2.rectangle(image, (obj.x_min, y - text_h), (obj.x_min + text_w, y), text_color_bg, -1)
        #cv2.rectangle(image, (obj.x_min, y), (obj.x_min + text_w, y + text_h), text_color_bg, -1)
        cv2.putText(image, label, (obj.x_min, y), font, font_scale, color_fg, font_thickness, cv2.LINE_AA)

    if False:
        detection_results = detector(img_attachment)
        detection_results.display(render=True)  # render detection boxes on image result
        predictions = detection_results.pred[0].tolist()
        #predictions is an array of results, with each result being an array with the following format:
        # subscript 0,1,2,3 = x1, x2, y1, y2 (box bounds)
        # subscript 4 = confidence (float between 0.0 and 1.0)
        # subscript 5 = category (numerical)
        found_objects = [(detector.names[int(e[5])], e[4]) for e in predictions]
        image = cv2.cvtColor(detection_results.imgs[0], cv2.COLOR_BGR2RGB)  # color out of yolov5 rendering is BGR, but needs to be RGB
    
    # print out all found objects
    for i, (category, confidence) in enumerate(found_objects):
        logger.info("worker{:02}: {}: '{}' identified with confidence of {:.2f}% ({})".format(
            thread_index,
            i, category, confidence * 100.0,
            'USING' if confidence >= conf['min_confidence']
                and category in conf['notify_on_dataset_categories'] else 'NOT USING'))
    
    # detection_results.t tuple is times for preprocessing, inferrance, and NMS
    logger.info("worker{:02}: Detection took {:.2f}ms".format(thread_index, (detection_end - detection_start) * 1000.0))

    found_objects_str = ', '.join([e[0] for e in found_objects])
    found_objects_str_with_confidence = ', '.join(['{} ({:.2f}%)'.format(e[0], e[1] * 100.0) for e in found_objects])

    # see if we should notify (based on found objects)
    to_notify = any([e[0] in conf['notify_on_dataset_categories'] for e in found_objects])
    if not to_notify and not always_notify:  #no recognitions over the confidence threshold
        if not found_objects:
            logger.info("worker{:02}: Not notifying as no found objects".format(thread_index))
        else:
            logger.info("worker{:02}: Not notifying as no suitable found objects -- FOUND: {} -- WANTED: {}".format(
                thread_index, found_objects_str, ', '.join(conf['notify_on_dataset_categories'])))
        return True

    # convert image over to a jpg string format
    successful_encode = False    
    for i, quality in enumerate([85, 50]):
        image_encode = cv2.imencode('.jpg', image,
            [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1, cv2.IMWRITE_JPEG_LUMA_QUALITY, quality])[1]
        image_str_encode = np.array(image_encode).tobytes()
        if len(image_str_encode) >= PUSHOVER_MAX_ATTACHMENT_SIZE:
            logger.warn("worker{:02}: Image size of {} too large for attachment with quality {} (max allowed: {})...".format(
                thread_index, util.convert_size(len(image_str_encode)), quality, util.convert_size(PUSHOVER_MAX_ATTACHMENT_SIZE)))
        else:
            successful_encode = True
            break
    if not successful_encode:
        logger.warn("worker{:02}: Image size still too large for attachment.".format(thread_index))
        return True
    logger.debug("worker{:02}: Resultant JPEG size: {}".format(thread_index, util.convert_size(len(image_str_encode))))

    # run platerecognizer on images that have vehicles
    run_platerecognizer = False
    platerecognizer_info = []
    for found_object in [e[0] for e in found_objects]:
        if found_object in PLATE_RECOGNIZER_ALLOWED_DETECTION_CATEGORIES:
            run_platerecognizer = True
            break
    if conf['platerecognizer_api_key'] and run_platerecognizer:
        #send the original image to platerecognizer, as any shading/flagging by object detection framework could reduce recognition accuracy
        img_attachment_raw.seek(0)  # just in case
        r = requests.post('https://api.platerecognizer.com/v1/plate-reader/', data=dict(regions=conf['platerecognizer_regions_hint']),
            files=dict(upload=img_attachment_raw), headers={'Authorization': 'Token ' + conf['platerecognizer_api_key']})
        if r.status_code not in (200, 201):
            logger.info("Invalid Platerecognizer response: {}".format(r.text))
            try:
                error_detail = r.json()['detail']
            except:
                error_detail = "UNKNOWN ERROR"
            platerecognizer_info.append("Platerecognizer API error: {}".format(error_detail))
        else:
            for vehicle in r.json()['results']:
                platerecognizer_info.append("Plate {}: {}, {}, conf {}".format(
                    vehicle['plate'], vehicle['region']['code'], vehicle['vehicle']['type'], vehicle['score']))

    # notify via pushover
    c = pushover.Client(conf['pushover_user_key'], api_token=conf['pushover_api_token'])
    c.send_message("Identified {}{}".format(found_objects_str_with_confidence,
        (' (' + ', '.join([vehicle for vehicle in platerecognizer_info]) + ')') if run_platerecognizer and platerecognizer_info else ''),
        title="{} Oddspot Alert".format(camera_name), attachment=('capture.jpg', image_str_encode))
    logger.info("worker{:02}: Alert sent via pushover".format(thread_index))

    # update state (will be dumped via atexit handlers)
    state['last_notify'][camera_name] = utc_now_epoch

    # signal processing of this queue item is done
    processing_queue.task_done()
    return True

def email_worker_loop(thread_index, processing_queue, detector):
    while email_worker_iter(thread_index, processing_queue, detector):
        if conf['memory_usage_logging']:
            #objgraph.show_most_common_types(limit=50)
            #objgraph.show_growth(limit=20)
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            logger.info("[ Top 10 ]")
            for stat in top_stats[:10]:
                logger.info(stat)

def terminate_workers_and_cleanup(logger, aiosmtpd_controller, num_worker_threads, processing_queue, signal=None, frame=None):
    # NOTE: we have this cleanup_called variable because of an odd situation where registering this via atexit.register
    # will not call it on SIGTERM (e.g. when shut down via `systemctl stop`), but when registering this via
    # signal.register for SIGTERM and atexit.register, this handler gets called two consecutive times at SIGTERM
    global cleanup_called

    if cleanup_called:
        return

    logger.debug("Program termination. Shutting down worker threads and cleaning up...(signal: {})".format(signal))
    if frame:
        logger.debug(traceback.format_stack(frame))

    for i in range(num_worker_threads):
        processing_queue.put(None)

    if aiosmtpd_controller:
        aiosmtpd_controller.stop()  # stop mail server if running
    if state:
        util.dump_state(STATE_FILE, state)

    cleanup_called = True

def main():
    global conf, state

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action='store_true', default=False, help="increase output verbosity")
    args = ap.parse_args()

    # set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    #stream_handler.setFormatter(formatter)
    file_handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1048576 * 2, backupCount=5)
    file_handler.setFormatter(formatter)
    # log both to file and to console
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info("START (uid: {}, gid: {})".format(os.geteuid(), os.getegid()))

    # set up exception and shutdown hooks
    sys.excepthook = handle_exception

    # load and validate config
    conf, cp = load_config()

    # if memory logging enabled, start that
    if conf['memory_usage_logging']:
        tracemalloc.start()

    # load state
    state = util.load_state(STATE_FILE)

    # set up processing queue for incoming mail messages
    processing_queue = queue.Queue()

    # start mail processing (smtpd) thread
    aiosmtpd_controller = AiosmtpdController(SMTPDHandler(processing_queue),
        hostname=conf['smtp_listen_host'], port=conf['smtp_listen_port'])
    logger.info("Starting mail server on host {}, port {}".format(
        'ALL' if conf['smtp_listen_host'] in (None, '') else conf['smtp_listen_host'], conf['smtp_listen_port']))
    aiosmtpd_controller.start()  # will start a separate thread
    logger.info("Mail server started")

    # determine # of worker threads to spawn
    # we support multi CUDA GPU use automatically. So if we detect multiple CUDA GPUs
    # available, start a cooresponding number of threads to feed each one
    num_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0  # TODO: support other GPU interfaces as well
    num_worker_threads = min(num_gpu, 1)  # if no GPUs, just 1 worker for the CPU
    logger.info("Detected {} CUDA GPUs".format(num_gpu))

    # wait on incoming mail messages
    logger.info("Starting {} worker threads...".format(num_worker_threads))
    threads = []
    for thread_index in range(num_worker_threads):
        # initialize detection engine
        logger.info("worker{:02}: Initializaing yolov5 detection engine... (thread_index: {})".format(thread_index, thread_index))
        
        # load yolov5 dataset on appropriate device
        #detector = torch.hub.load('ultralytics/yolov5', conf['yolov5_model'], device=torch.device(thread_index))
        #logging.getLogger('yolov5').setLevel(logging.INFO)
        # start thread for processing with this model
        #t = threading.Thread(target=email_worker_loop, args=(thread_index, processing_queue, detector))

        deepstack_config = ServerConfig("http://localhost:{}".format(conf['deepstack_api_port']))
        detector = Detection(deepstack_config)

        t = threading.Thread(target=email_worker_loop, args=(thread_index, processing_queue, detector))
        t.start()
        threads.append(t)
    logger.info("Started {} worker threads {}".format(num_worker_threads, threads))

    # register via signal.signal for SIGTERM (e.g. via `systemctl stop`) as well as non-SIGTERM (via atexit.register)
    signal.signal(signal.SIGTERM, lambda signal, frame: terminate_workers_and_cleanup(
        logger, aiosmtpd_controller, num_worker_threads, processing_queue, signal=signal, frame=frame))
    atexit.register(terminate_workers_and_cleanup, logger, aiosmtpd_controller, num_worker_threads, processing_queue)

    # Wait for all worker threads to finish (will normally sit forever,
    # until exit signal/keyboard interrupt/exception received)
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        for i in range(num_worker_threads):
            processing_queue.put(None)
        
if __name__ == "__main__":
    main()
        
