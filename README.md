# oddspot
Uses MobileNet NN to locate and notify security cam images based on presence of people, vehicles

## Credits
Neural net code originally taken from https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/
Email parsing originally from https://www.ianlewis.org/en/parsing-email-attachments-python

## Dependencies

* Python 3
* Postfix
* `pip3 install python-pushover numpy opencv-contrib-python-headless`

## Sample config

Create a file `oddspot.ini` within the `oddspot` base directory containing the following:

```
[Default]
pushover_user_key=<YOUR USER KEY HERE>
pushover_api_token=<YOUR API KEY HERE>
min_confidence=0.7
model_prototxt_file=MobileNetSSD_deploy.prototxt.txt
model_caffe_file=MobileNetSSD_deploy.caffemodel
model_classes=background,bus,car,motorbike,person
min_notify_period=600
```

## Setup

These instructions are for Ubuntu 18.04 LTS using a factory Postfix install.

Edit your `/etc/aliases` and add the following line (replacing the path as necessary):
```
oddspot:  "|/YOUR/PATH/TO/oddspot.py"
```

Then, run `sudo newaliases`.

## Usage

Modify your camera setup (via its embedded web configuration page) to send an email to `oddspot@<YOURMACHINE>` on motion detection events. Make sure it attaches a snapshot of the event in jpeg or png format (and _not_ a video). Also, to avoid flooding the script, you should probably have it so that it will only send repeat emails every minute or more. The `oddspot` script allows you to further limit continuous emails on a per camera basis via the `min_notify_period` config setting, but it should get raw emails from the cameras on motion events more frequently than that, to maximize alerting accuracy).
