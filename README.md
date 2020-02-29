# oddspot
Uses MobileNet NN to locate and notify security cam images based on presence of people, vehicles

## Credits

* Neural net code originally taken from [here](https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/).
* Email parsing originally from [here](https://www.ianlewis.org/en/parsing-email-attachments-python).

## Setup (Ubuntu 18.04)

These instructions are for a stock Ubuntu 18.04 LTS system.

Install dependencies:
```
sudo apt-get -y install postfix gcc g++ python3 python3-setuptools mailutils
sudo pip3 install python-pushover numpy opencv-contrib-python-headless torch torchvision
sudo pip3 install cython; sudo pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Set up Postfix to run our script on an incoming email to `oddspot@<OUR SERVER>` (**change the text surrounded by `<>` (e.g. `<USER>`) as appropriate**):
```
sudo bash -c 'echo "oddspot: \"|sudo -u <USER> /home/<USER>/oddspot/oddspot.py --debug\"" >> /etc/aliases'
sudo newaliases
sudo bash -c 'echo "nobody ALL=(<USER>) NOPASSWD: /home/<USER>/oddspot/oddspot.py --debug" > /etc/sudoers.d/oddspot'
```

## Sample config

Create a file `oddspot.ini` within the `oddspot` base directory containing the following:

```
[Default]
pushover_user_key=<YOUR USER KEY HERE>
pushover_api_token=<YOUR API KEY HERE>

#min_notify_period: minimum object identification confidence level in order to notify
# we will notify if any identified object in the image has >= this level
# e.g. 0.2 = 20%, 0.7 = 70%
min_confidence=0.7

#min_notify_period: only send out a pushover notification for a given camera this often (in seconds)
min_notify_period=600

#camera_names: a JSON object that maps the From email address of the sending camera to a name that will show in the notification for it
camera_names={"local@smtp01.localnet": "testcam", "root@cam-front.localnet": "cam-front", "root@cam-back.localnet": "cam-back"}

#objdetection_framework: either detectron2 or opencv_mobilenetssd
objdetection_framework=detectron2

#notify_dataset_classes: object labels/classes to notify on
notify_dataset_classes=bus,car,motorcycle,person

#detectron2_config_file: detectron2 config file to use.
# See options at: https://github.com/facebookresearch/detectron2/tree/master/configs
#detectron2_config_file=configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml

#detectron2_extra_opts: Extra configuration options sent to detectron2
# defaults to: "MODEL.DEVICE cpu"
# Uncomment the line below to run on an NVIDIA GPU via CUDA
#detectron2_extra_opts="MODEL.DEVICE cuda"
# (Other GPU options are: mkldnn, opengl, opencl, ideep, hip, msnpu)
```

## Testing

* Copy a camera capture image to the system, and run the following command to send a test email that Postfix should pick up and process:

`echo "Test email body" | mail -s "Test 123" oddspot@<SERVER HOSTNAME> -A <TESTIMAGE>.jpg`

Tail `/var/log/mail.log` and `/home/<USER>/oddspot/logs/oddspot.log` to see if processing was successful or not.

## Usage

Modify your camera setup (via its embedded web configuration page) to send an email to `oddspot@<YOURMACHINE>` on motion detection events. Make sure it attaches a snapshot of the event in jpeg or png format (and _not_ a video). Also, to avoid flooding the script, you should probably have it so that it will only send repeat emails every minute or more. The `oddspot` script allows you to further limit continuous emails on a per camera basis via the `min_notify_period` config setting, but it should get raw emails from the cameras on motion events more frequently than that, to maximize alerting accuracy).
