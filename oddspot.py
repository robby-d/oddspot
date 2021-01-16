#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
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
from builtins import str

from memory_profiler import profile
import objgraph

import pushover
import torch
import cv2
import numpy as np
from aiosmtpd.controller import Controller as AiosmtpdController

import util
from objdetection import detectron2, opencv_mobilenetssd

# memory leak tracing
import tracemalloc
tracemalloc.start()

PROG_NAME = "oddspot"
CURDIR = os.path.dirname(os.path.realpath(__file__))
STATE_FILE = os.path.join(CURDIR, "{}.dat".format(PROG_NAME))
LOG_FILE = os.path.join(CURDIR, "logs", "{}.log".format(PROG_NAME))
CONF_FILE = os.path.join(CURDIR, "{}.ini".format(PROG_NAME))

DEFAULT_MODEL_CLASSES_OVERRIDE = []
DEFAULT_MIN_CONFIDENCE = 0.7  # 70%
OBJDETECTION_FRAMEWORK_CHOICES = ("detectron2", "opencv_mobilenetssd")
DEFAULT_OBJDETECTION_FRAMEWORK = "detectron2"
DEFAULT_MODEL_CLASSES = "background,car,bus,person"
DEFAULT_MIN_NOTIFY_PERIOD = 600  # in seconds (600 = 10 minutes)
PUSHOVER_MAX_ATTACHMENT_SIZE = 2621440  # 2.5MB

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

    conf['objdetection_framework'] = cp.get('detection', 'objdetection_framework', fallback=DEFAULT_OBJDETECTION_FRAMEWORK)
    if conf['objdetection_framework'] not in OBJDETECTION_FRAMEWORK_CHOICES:
        raise Exception("'objdetection_framework' config option must be one of: {}".format(', '.join(OBJDETECTION_FRAMEWORK_CHOICES)))

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

    conf['sender_camera_names'] = cp.get('smtpd', 'sender_camera_names', fallback={})
    if conf['sender_camera_names']:
        conf['sender_camera_names'] = json.loads(conf['sender_camera_names'])
    assert isinstance(conf['sender_camera_names'], dict)

    # section: integrations
    conf['platerecognizer_api_key'] = cp.get('integrations', 'platerecognizer_api_key').strip()
    conf['platerecognizer_regions_hint'] = cp.get('integrations', 'platerecognizer_regions_hint')
    if conf['platerecognizer_regions_hint']:
        conf['platerecognizer_regions_hint'] = json.loads(conf['platerecognizer_regions_hint'])
    else:
        conf['platerecognizer_regions_hint'] = []
    assert isinstance(conf['platerecognizer_regions_hint'], (list, tuple))

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
        if address.lower() != 'oddspot@localhost':
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
    img_attachment = parsed_email['attachments'][0]
    img_attachment.seek(0)  # just in case

    #perform detection
    if conf['objdetection_framework'] == 'detectron2':
        found_objects, image = detector.do_detections(img_attachment)
    else:
        assert conf['objdetection_framework'] == 'opencv_mobilenetssd'
        found_objects, image = detector.do_detections(conf['min_confidence'], img_attachment)

    # print out all found objects
    for i, (category, confidence) in enumerate(found_objects):
        logger.info("worker{:02}: {}: '{}' identified with confidence of {:.2f}% ({})".format(
            thread_index,
            i, category, confidence * 100.0,
            'USING' if confidence >= conf['min_confidence']
                and category in conf['notify_on_dataset_categories'] else 'NOT USING'))

    found_objects_str = ', '.join([e[0] for e in found_objects])
    found_objects_str_with_confidence = ', '.join(['{} ({:.2f}%)'.format(e[0], e[1] * 100.0) for e in found_objects])

    # see if we should notify
    utc_now = datetime.datetime.utcnow()
    utc_now_epoch_timestamp = utc_now.timestamp()
    utc_now_epoch = int(utc_now_epoch_timestamp)  # second precision
    to_notify = any([e[0] in conf['notify_on_dataset_categories'] for e in found_objects])
    if not to_notify:  #no recognitions over the confidence threshold
        if not found_objects:
            logger.info("worker{:02}: Not notifying as no found objects".format(thread_index))
        else:
            logger.info("worker{:02}: Not notifying as no suitable found objects -- FOUND: {} -- WANTED: {}".format(
                thread_index, found_objects_str, ', '.join(conf['notify_on_dataset_categories'])))
        return True

    camera_name = conf['sender_camera_names'].get(parsed_email['from'], parsed_email['from'])
    state['last_notify'].setdefault(camera_name, False)
    assert state['last_notify'][camera_name] is False or utc_now_epoch - state['last_notify'][camera_name] >= 0
    if state['last_notify'][camera_name] is not False and utc_now_epoch - state['last_notify'][camera_name] < conf['min_notify_period']:
        logger.info("worker{:02}: Not notifying as last notification for this camera is {} seconds ago (needs to be >= {} seconds)".format(
            thread_index, utc_now_epoch - state['last_notify'][camera_name], conf['min_notify_period']))
        return True

    # convert image over to a jpg string format
    successful_encode = False    
    for i, quality in enumerate([85, 50]):
        image_encode = cv2.imencode('.jpg', image,
            [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1, cv2.IMWRITE_JPEG_LUMA_QUALITY, quality])[1]
        image_str_encode = np.array(image_encode).tostring()
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
        if found_object in ('car', 'truck', 'bus'):
            run_platerecognizer = True
            break
    if conf['platerecognizer_api_key'] and run_platerecognizer:
        r = requests.post('https://api.platerecognizer.com/v1/plate-reader/', data=dict(regions=conf['platerecognizer_regions_hint']),
            files=dict(upload=image_str_encode), headers={'Authorization': 'Token ' + conf['platerecognizer_api_key']})
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
        #objgraph.show_most_common_types(limit=50)
        #objgraph.show_growth(limit=20)
        #pass

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
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1048576 * 2, backupCount=5)
    file_handler.setFormatter(formatter)
    # log both to file and to console
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())
    logger.info("START")

    # set up exception and shutdown hooks
    sys.excepthook = handle_exception

    # load and validate config
    conf, cp = load_config()

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
    # with detectron2, we support multi CUDA GPU use automatically. So if we detect multiple CUDA GPUs
    # available, start a cooresponding number of threads to feed each one
    num_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0  # TODO: support other GPU interfaces as well
    if conf['objdetection_framework'] == 'detectron2':
        num_worker_threads = min(num_gpu, 1)  # if no GPUs, just 1 worker for the CPU
    else:
        num_worker_threads = 1
    logger.info("Detected {} CUDA GPUs".format(num_gpu))

    # wait on incoming mail messages
    logger.info("Starting {} worker threads...".format(num_worker_threads))
    threads = []
    for thread_index in range(num_worker_threads):
        # initialize detection engine
        logger.info("worker{:02}: Initializaing {} detection engine... (thread_index: {})".format(thread_index, conf['objdetection_framework'], thread_index))
        if conf['objdetection_framework'] == 'detectron2':
            detector = detectron2.Detector(conf, cp, gpu_id=thread_index+1 if num_gpu else 0)
        else:
            assert conf['objdetection_framework'] == 'opencv_mobilenetssd'
            detector = opencv_mobilenetssd.Detector(conf, cp)

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
        