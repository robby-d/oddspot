# -*- coding: utf-8 -*-

import sys
import os
import io
import math
import pickle
import logging

logger = logging.getLogger(__name__)

class EmptyStateFileException(Exception):
    pass

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

def load_state(statefile_path):
    try:
        state = pickle.load(open(statefile_path, "rb"))
        if not state:  # empty state files
            raise EmptyStateFileException
        logger.debug("Loaded state: %s" % state)
    except (FileNotFoundError, EmptyStateFileException) as e:
        logger.warning("No state file")
        state = {
            'last_notify': {},
            #^  key = camera name, value = epoch int of when we last sent out a pushover for that camera
        }
    return state

def dump_state(statefile_path, state):
    logger.debug("Dumping state: %s" % state)
    pickle.dump(state, open(statefile_path, "wb"))

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
