# oddspot
Uses MobileNet NN to locate and notify security cam images based on presence of people, vehicles

## Credits

* Neural net code originally taken from [here](https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/).
* Email parsing originally from [here](https://www.ianlewis.org/en/parsing-email-attachments-python).

## Setup (Ubuntu 20.04 LTS)

These instructions are for a stock Ubuntu 20.04 LTS system.

Install dependencies:
```
sudo apt-get -y install gcc g++ python3 python3-setuptools sendemail
sudo pip3 install python-pushover numpy opencv-contrib-python-headless torch torchvision aiosmtpd
sudo pip3 install cython; sudo pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
sudo pip3 install 'git+https://github.com/facebookresearch/detectron2.git'
```

If you are doing analysis with a NVIDIA GPU, also run this command to install the CUDA toolkit:
```
sudo apt-get -y install nvidia-cuda-toolkit
```

## Sample config

Create a file `oddspot.ini` within the `oddspot` base directory containing the following:

```
[detection]
#min_notify_period: minimum object identification confidence level in order to notify
# we will notify if any identified object in the image has >= this level
# e.g. 0.2 = 20%, 0.7 = 70%
min_confidence=0.7

#objdetection_framework: either detectron2 or opencv_mobilenetssd
objdetection_framework=detectron2

#detectron2_config_file: detectron2 config file to use.
# See options at: https://github.com/facebookresearch/detectron2/tree/master/configs
#detectron2_config_file=configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml

#detectron2_extra_opts: Extra configuration options sent to detectron2
# Uncomment the line below to run on an opengl GPU interface
#detectron2_extra_opts=MODEL.DEVICE opengl
# (GPU options are: cuda, mkldnn, opengl, opencl, ideep, hip, msnpu)
# NOTE that oddspot will auto detect available CUDA GPUs and parallelize across them
# Other GPU types require manual configuration via this option

[notify]
pushover_user_key=<YOUR USER KEY HERE>
pushover_api_token=<YOUR API KEY HERE>

#min_notify_period: only send out a pushover notification for a given camera this often (in seconds)
min_notify_period=600

#notify_on_dataset_categories: object labels/classes to notify on
notify_on_dataset_categories=bus,car,motorcycle,person,truck

[smtpd]
listen_host =
listen_port = 10025

[integrations]
platerecognizer_api_key=<YOUR PLATERECOGNIZER.COM API KEY HERE OR BLANK TO DISABLE>
#platerecognizer_regions_hint: array of platerecognizer reagons codes to provide as a hint to the object recognizer (blank or empty array to disable)
platerecognizer_regions_hint=["us-nc", "us-va"]

[cameras]
#camera_names_from_sender: a JSON object that maps the From email address of the sending camera to a name that will show in the notification for it
camera_names_from_sender={"local@smtp01.localnet": "testcam", "root@cam-front.localnet": "cam-front", "root@cam-back.localnet": "cam-back"}
#camera_custom_configs: a JSON dict of dicts -- totally optional. Main dict keys are camera names. For each supplied camera name, certain options can be specified
# currently available options:
#   - always_notify: specify as true to always notify, regardless of image analysis results -- useful if the camera in question does its own advanced analysis
camera_custom_configs={"testcam": {"always_notify": true}}
```

## Testing

Start the command interactively, as any user account you wish. (Obviously, if you have your smtp port set to <= 1024, the service will need to run as `root`.):

`cd /home/<USER>/oddspot; ./oddspot.py --debug`

Then, copy a camera capture image to the system, and run the following command in another terminal window to send a test email that `oddspot` should pick up and process:

`sendemail -f test@test.com -t oddspot@localhost -s localhost:10025 -u 'testing 123' -m 'test' -a <TESTIMAGE>.jpg`

Tail `/home/<USER>/oddspot/logs/oddspot.log` to see if processing was successful or not.

## Usage

If testing was successful, you can have the `oddpspot` daemon start on startup via creating a file at `/etc/systemd/system/oddspot.service` with the following contents:

```
[Unit]
Description=oddspot

[Service]
ExecStart=/home/<USER>/oddspot/oddspot.py
User=<USER>
Restart=on-failure
RestartSec=30s

[Install]
WantedBy=multi-user.target
```

Then run:

```
sudo systemctl enable oddspot
sudo systemctl start oddspot
```

Modify your camera setup (via its embedded web configuration page) to send an email to `oddspot@<YOURMACHINE>:<YOURPORT>` on motion detection events. Make sure it attaches a snapshot of the event in jpeg or png format (and _not_ a video). Also, to avoid flooding the script, you should probably have it so that it will only send repeat emails every minute or more. The `oddspot` script allows you to further limit continuous emails on a per camera basis via the `min_notify_period` config setting, but it should get raw emails from the cameras on motion events more frequently than that, to maximize alerting accuracy).


## Usage with a CUDA capable GPU

If you have a CUDA-capable NVIDIA GPU installed in your system, `oddspot` can work with it if using the `detectron2` framework. 

If using Ubuntu, install the CUDA drivers:
```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-driver-460
```

Add (or uncomment) the following line in your `oddspot.ini` file:
```
detectron2_extra_opts=MODEL.DEVICE cuda
```

Then reboot your system for the driver to be properly loaded.
