#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import the necessary packages
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

import numpy as np
import cv2
import pushover

PROG_NAME = "oddspot"
CURDIR = os.path.dirname(os.path.realpath(__file__))
STATE_FILE = os.path.join(CURDIR, "{}.dat".format(PROG_NAME))
LOG_FILE = os.path.join(CURDIR, "logs", "{}.log".format(PROG_NAME))
CONF_FILE = os.path.join(CURDIR, "{}.ini".format(PROG_NAME))

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
MODEL_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
MODEL_COLORS = np.random.uniform(0, 255, size=(len(MODEL_CLASSES), 3))
DEFAULT_MIN_CONFIDENCE = 0.7  # 70%
DEFAULT_MODEL_PROTOTXT_FILE = "MobileNetSSD_deploy.prototxt.txt"
DEFAULT_MODEL_CAFFE_FILE = "MobileNetSSD_deploy.caffemodel"
DEFAULT_MODEL_CLASSES = "background,car,bus,person"
DEFAULT_MIN_NOTIFY_PERIOD = 600  # in seconds (600 = 10 minutes)

logger = logging.getLogger(__name__)
utc_now = datetime.datetime.utcnow()
utc_now_epoch_timestamp = utc_now.timestamp()
utc_now_epoch = int(utc_now_epoch_timestamp)  # second precision

def load_state():
    try:
        state = pickle.load(open(STATE_FILE, "rb"))
        if not state:  # empty state file
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
            attachment = email_parse_multipart_attachment(part, multipart=True)
            if attachment:
                attachments.append(attachment)
            elif part.get_content_type() == "text/plain":
                if body is None:
                    body = ""
                body += unicode(
                    part.get_payload(decode=True),
                    part.get_content_charset(),
                    'replace'
                ).encode('utf8','replace')
            elif part.get_content_type() == "text/html":
                if html is None:
                    html = ""
                html += unicode(
                    part.get_payload(decode=True),
                    part.get_content_charset(),
                    'replace'
                ).encode('utf8','replace')

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
    stream_handler = logging.StreamHandler()
    #stream_handler.setFormatter(formatter)
    # log both to file and to console
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
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
    if not os.path.exists(CONF_FILE):
        raise Exception("Config file does not exist at path: '{}'".format(CONF_FILE))
    conf = {}
    config = configparser.ConfigParser()
    config.read(CONF_FILE)
    conf['pushover_user_key'] = config.get('Default', 'pushover_user_key')
    conf['pushover_api_token'] = config.get('Default', 'pushover_api_token')
    conf['min_confidence'] = config.getfloat('Default', 'min_confidence', fallback=DEFAULT_MIN_CONFIDENCE)
    assert conf['min_confidence'] > 0.0 and conf['min_confidence'] <= 1.0
    conf['model_prototxt_file'] = config.get('Default', 'model_prototxt_file', fallback=DEFAULT_MODEL_PROTOTXT_FILE)
    model_prototxt_file_path = os.path.join(CURDIR, 'models', conf['model_prototxt_file'])
    if not os.path.exists(model_prototxt_file_path):
        raise Exception("Could not find specified model prototxt file: '{}'".format(model_prototxt_file_path))
    conf['model_caffe_file'] = config.get('Default', 'model_caffe_file', fallback=DEFAULT_MODEL_CAFFE_FILE)
    model_caffe_file_path = os.path.join(CURDIR, 'models', conf['model_caffe_file'])
    if not os.path.exists(model_caffe_file_path):
        raise Exception("Could not find specified model caffe file: '{}'".format(model_caffe_file_path))
    conf['model_classes'] = config.get('Default', 'model_classes', fallback=DEFAULT_MODEL_CLASSES)
    conf['model_classes'] = [e.strip() for e in conf['model_classes'].split(',')]
    if 'background' not in conf['model_classes']:
        raise Exception("'background' must be listed in supplied model_classes config value")
    for e in conf['model_classes']:
        if e not in MODEL_CLASSES:
            raise Exception("Invalid value '{}' is listed in supplied model_classes config value. Valid options: {}".format(
                e, ', '.join(MODEL_CLASSES)))
    conf['min_notify_period'] = config.getint('Default', 'min_notify_period', fallback=DEFAULT_MIN_NOTIFY_PERIOD)
    assert conf['min_notify_period'] >= 0
    conf['camera_names'] = config.get('Default', 'camera_names', fallback={})
    if conf['camera_names']:
        conf['camera_names'] = json.loads(conf['camera_names'])
    assert isinstance(conf['camera_names'], dict)

    # load state
    state = load_state()

    # get email message data from stdin
    stdin_data = sys.stdin.read()
    parsed_email = email_parse(stdin_data)

    # parse the raw image data out of the email
    logger.debug("Email from: '{}', subject: '{}'".format(parsed_email['from'], parsed_email['subject']))
    logger.debug("Email has {} attachments: {}".format(len(parsed_email['attachments']),
        ', '.join([a.content_type for a in parsed_email['attachments']]) if len(parsed_email['attachments']) else 'N/A'))
    if not parsed_email['attachments'] or parsed_email['attachments'][0].content_type not in ('application/octet-stream', 'image/jpeg', 'image/png'):
        raise Exception("Cannot parse out image from stdin email")
    img_attachment = parsed_email['attachments'][0]
    logger.info("Using attachment 0, type: {}, size: {}".format(img_attachment.content_type, img_attachment.size))

    # load our serialized model from disk
    logger.info("loading model {} (prototxt: {})...".format(conf['model_caffe_file'], conf['model_prototxt_file']))
    net = cv2.dnn.readNetFromCaffe(model_prototxt_file_path, model_caffe_file_path)

    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    # (note: normalization is done via the authors of the MobileNet SSD
    # implementation)
    img_attachment.seek(0)  # just in case
    bytes_as_np_array = np.frombuffer(img_attachment.read(), dtype=np.uint8)
    image = cv2.imdecode(bytes_as_np_array, cv2.IMREAD_UNCHANGED)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and predictions
    logger.debug("computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    found_objects = []
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        idx = int(detections[0, 0, i, 1])
        if MODEL_CLASSES[idx] != 'background':
            logger.debug("Initial object {} identified with confidence of {:.2f}%".format(
                MODEL_CLASSES[idx], confidence * 100))
        if confidence > conf['min_confidence']:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            #idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(MODEL_CLASSES[idx], confidence * 100)
            found_objects.append((MODEL_CLASSES[idx], confidence * 100))
            logger.info("TAGGED: {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY),
                MODEL_COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, MODEL_COLORS[idx], 2)

    # see if we should notify
    if not any([e[0] in conf['model_classes'] and e[0] != 'background' for e in found_objects]):
        logger.info("Not notifying as found classes do not match what we are looking for (found: {}, looking for: {})".format(
            ', '.format([e[0] for e in found_classes]), ', '.format(conf['model_classes'])))
        sys.exit(0)

    camera_name = conf['camera_names'].get(parsed_email['from'], parsed_email['from'])
    state['last_notify'].setdefault(camera_name, 0)
    assert utc_now_epoch - state['last_notify'][camera_name] >= 0
    if utc_now_epoch - state['last_notify'][camera_name] < conf['min_notify_period']:
        logger.info("Not notifying last notification for this camera is {} seconds ago (needs to be >= {} seconds)".format(
            utc_now_epoch - state['last_notify'][camera_name], conf['min_notify_period']))
        sys.exit(0)

    image_encode = cv2.imencode('.jpg', image)[1]
    str_encode = np.array(image_encode).tostring()

    # notify via pushover
    c = pushover.Client(conf['pushover_user_key'], api_token=conf['pushover_api_token'])
    found_objects_str = ', '.join(['{} ({:.2f}%)'.format(e[0], e[1]) for e in found_objects])
    c.send_message("Identified {}".format(found_objects_str), title="{} Oddspot Alert".format(camera_name), attachment=('capture.jpg', str_encode))
    logger.info("Sent image (size: {} bytes) via pushover".format(len(str_encode)))

    # update state
    state['last_notify'][camera_name] = utc_now_epoch
    if camera_name in state:
        del state[camera_name]

    # dump state
    dump_state(state)


if __name__ == "__main__":
    main()