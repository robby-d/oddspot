#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import io
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
import math
from builtins import str

import pushover
import cv2
import numpy as np


from objdetection import detectron2, opencv_mobilenetssd

PROG_NAME = "oddspot"
CURDIR = os.path.dirname(os.path.realpath(__file__))
STATE_FILE = os.path.join(CURDIR, "{}.dat".format(PROG_NAME))
LOG_FILE = os.path.join(CURDIR, "logs", "{}.log".format(PROG_NAME))
CONF_FILE = os.path.join(CURDIR, "{}.ini".format(PROG_NAME))

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
DEFAULT_MODEL_CLASSES_OVERRIDE = []
DEFAULT_MIN_CONFIDENCE = 0.7  # 70%
OBJDETECTION_FRAMEWORK_CHOICES = ("detectron2", "opencv_mobilenetssd")
DEFAULT_OBJDETECTION_FRAMEWORK = "detectron2"
DEFAULT_MODEL_CLASSES = "background,car,bus,person"
DEFAULT_MIN_NOTIFY_PERIOD = 600  # in seconds (600 = 10 minutes)

PUSHOVER_MAX_ATTACHMENT_SIZE = 2621440  #2.5MB

logger = logging.getLogger(__name__)
utc_now = datetime.datetime.utcnow()
utc_now_epoch_timestamp = utc_now.timestamp()
utc_now_epoch = int(utc_now_epoch_timestamp)  # second precision

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

def load_state():
    try:
        state = pickle.load(open(STATE_FILE, "rb"))
        if not state:  # empty state files
            raise Exception
        logger.debug("Loaded state: %s" % state)
    except:
        logger.warning("No state file")
        state = {
            'last_notify': {},
            #^  key = camera name, value = epoch int of when we last sent out a pushover for that camera
        }
    return state

def dump_state(state):
    logger.debug("Dumping state: %s" % state)
    pickle.dump(state, open(STATE_FILE, "wb"))

def load_config():
    if not os.path.exists(CONF_FILE):
        raise Exception("Config file does not exist at path: '{}'".format(CONF_FILE))
    conf = {}
    cp = configparser.ConfigParser()
    cp.read(CONF_FILE)

    conf['pushover_user_key'] = cp.get('Default', 'pushover_user_key')
    conf['pushover_api_token'] = cp.get('Default', 'pushover_api_token')

    conf['min_confidence'] = cp.getfloat('Default', 'min_confidence', fallback=DEFAULT_MIN_CONFIDENCE)
    assert conf['min_confidence'] > 0.0 and conf['min_confidence'] <= 1.0

    conf['objdetection_framework'] = cp.get('Default', 'objdetection_framework', fallback=DEFAULT_OBJDETECTION_FRAMEWORK)
    if conf['objdetection_framework'] not in OBJDETECTION_FRAMEWORK_CHOICES:
        raise Exception("'objdetection_framework' config option must be one of: {}".format(', '.join(OBJDETECTION_FRAMEWORK_CHOICES)))

    conf['min_notify_period'] = cp.getint('Default', 'min_notify_period', fallback=DEFAULT_MIN_NOTIFY_PERIOD)
    assert conf['min_notify_period'] >= 0

    conf['notify_on_dataset_categories'] = cp.get('Default', 'notify_on_dataset_categories')
    conf['notify_on_dataset_categories'] = [e.strip() for e in conf['notify_on_dataset_categories'].split(',')]

    conf['camera_names'] = cp.get('Default', 'camera_names', fallback={})
    if conf['camera_names']:
        conf['camera_names'] = json.loads(conf['camera_names'])
    assert isinstance(conf['camera_names'], dict)

    return conf, cp

def email_parse_attachment(message_part, multipart=True):
    if multipart:
        content_disposition = message_part.get("Content-Disposition", None)
        if not content_disposition:
            return None
        dispositions = content_disposition.strip().split(";")
        if dispositions[0].lower() != "attachment":
            return None

    file_data = message_part.get_payload(decode=True)
    attachment = io.BytesIO(file_data)
    attachment.content_type = message_part.get_content_type()
    attachment.size = len(file_data)
    attachment.name = None
    attachment.create_date = None
    attachment.mod_date = None
    attachment.read_date = None
    return attachment

def email_parse(content):
    from email.header import decode_header
    from email.parser import Parser as EmailParser
    from email.utils import parseaddr

    p = EmailParser()
    msgobj = p.parsestr(content)

    #f = open('/tmp/email.raw', 'w')
    #f.write(content)
    #f.close()

    if msgobj['Subject'] is not None:
        decodefrag = decode_header(msgobj['Subject'])
        subj_fragments = []
        for s , enc in decodefrag:
            if enc:
                s = unicode(s , enc).encode('utf8','replace')
            subj_fragments.append(s)
        subject = ''.join(subj_fragments)
    else:
        subject = None

    attachments = []
    body = None
    html = None

    if not msgobj.is_multipart():
        attachment = email_parse_attachment(msgobj, multipart=False)
        if attachment:
            attachments.append(attachment)
    else:
        for part in msgobj.walk():
            attachment = email_parse_attachment(part, multipart=True)
            if attachment:
                attachments.append(attachment)
            elif part.get_content_type() == "text/plain":
                if body is None:
                    body = ""
                body += part.get_payload(decode=True).decode('utf8', 'replace')
            elif part.get_content_type() == "text/html":
                if html is None:
                    html = ""
                html += part.get_payload(decode=True).decode('utf8', 'replace')

    return {
        'subject' : subject,
        'body' : body,
        'html' : html,
        'from' : parseaddr(msgobj.get('From'))[1],
        'to' : parseaddr(msgobj.get('To'))[1],
        'attachments': attachments,
    }

def main():
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

    # set up exception hook
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
    sys.excepthook = handle_exception

    # load and validate config
    conf, cp = load_config()

    # load state
    state = load_state()

    # get email message data from stdin
    stdin_data = sys.stdin.read()
    parsed_email = email_parse(stdin_data)

    # parse the raw image data out of the email
    logger.info("Email from: '{}', subject: '{}'".format(parsed_email['from'], parsed_email['subject']))
    logger.info("Email has {} attachments: {}".format(len(parsed_email['attachments']),
        ', '.join(['{}: {} (size: {}) ({})'.format(i, a.content_type, convert_size(a.size), 'USING' if i==0 else 'NOT USING')
            for i, a in enumerate(parsed_email['attachments'])]) if len(parsed_email['attachments']) else 'N/A'))
    if not parsed_email['attachments'] or parsed_email['attachments'][0].content_type not in ('application/octet-stream', 'image/jpeg', 'image/png'):
        raise Exception("Cannot parse out image from stdin email")
    # use first attachment
    img_attachment = parsed_email['attachments'][0]
    img_attachment.seek(0)  # just in case

    if conf['objdetection_framework'] == 'detectron2':
        found_objects, image = detectron2.do_detections(conf, cp, img_attachment)
    else:
        assert conf['objdetection_framework'] == 'opencv_mobilenetssd'
        found_objects, image = opencv_mobilenetssd.do_detections(conf, cp, img_attachment)

    # print out all found objects
    for i, (category, confidence) in enumerate(found_objects):
        logger.info("{}: '{}' identified with confidence of {:.2f}% ({})".format(
            i, category, confidence * 100,
            'USING' if confidence >= conf['min_confidence']
                and category in conf['notify_on_dataset_categories'] else 'NOT USING'))

    # see if we should notify
    to_notify = any([e[0] in conf['notify_on_dataset_categories'] for e in found_objects])
    if not to_notify:  #no recognitions over the confidence threshold
        if not found_objects:
            logger.info("Not notifying as no found objects")
        else:
            logger.info("Not notifying as found objects not what we are looking for -- FOUND: {} -- WANTED: {}".format(
                ', '.join([e[0] for e in found_objects]), ', '.join(conf['notify_on_dataset_categories'])))
        sys.exit(0)

    camera_name = conf['camera_names'].get(parsed_email['from'], parsed_email['from'])
    state['last_notify'].setdefault(camera_name, 0)
    assert utc_now_epoch - state['last_notify'][camera_name] >= 0
    if utc_now_epoch - state['last_notify'][camera_name] < conf['min_notify_period']:
        logger.info("Not notifying last notification for this camera is {} seconds ago (needs to be >= {} seconds)".format(
            utc_now_epoch - state['last_notify'][camera_name], conf['min_notify_period']))
        sys.exit(0)

    # convert image over to a jpg string format
    successful_encode = False    
    for i, quality in enumerate([85, 50]):
        image_encode = cv2.imencode('.jpg', image,
            [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1, cv2.IMWRITE_JPEG_LUMA_QUALITY, quality])[1]
        image_str_encode = np.array(image_encode).tostring()
        if len(image_str_encode) >= PUSHOVER_MAX_ATTACHMENT_SIZE:
            logger.warn("Image size of {} too large for attachment with quality {} (max allowed: {})...".format(
                convert_size(len(image_str_encode)), quality, convert_size(PUSHOVER_MAX_ATTACHMENT_SIZE)))
        else:
            successful_encode = True
            break
    if not successful_encode:
        raise Exception("Image size still too large for attachment.")
    logger.debug("Resultant JPEG size: {}".format(convert_size(len(image_str_encode))))

    # notify via pushover
    c = pushover.Client(conf['pushover_user_key'], api_token=conf['pushover_api_token'])
    found_objects_str = ', '.join(['{} ({:.2f}%)'.format(e[0], e[1]) for e in found_objects])
    c.send_message("Identified {}".format(found_objects_str),
        title="{} Oddspot Alert".format(camera_name), attachment=('capture.jpg', image_str_encode))
    logger.info("Alert sent via pushover")

    # update state
    state['last_notify'][camera_name] = utc_now_epoch

    # dump state
    dump_state(state)


if __name__ == "__main__":
    main()
