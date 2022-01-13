# oddspot
Daemon that listens for emails from cameras (with motion detection images attached) and uses [Deepstack](https://deepstack.cc/) to identify the presence of people, vehicles, et cetra and notify their existance via [Pushover](https://pushover.net/).

## Setup (Ubuntu 20.04 LTS)

These instructions are for a stock Ubuntu 20.04 LTS system.

Install base dependencies:
```
sudo apt-get update
sudo apt-get -y install gcc g++ python3 python3-setuptools sendemail docker.io
sudo pip3 install python-pushover aiosmtpd deepstack-sdk
```

If you have a CUDA-capable NVIDIA GPU installed in your system, `oddspot` will work with it automatically, as long as the drivers are installed. 
```
sudo apt-get -y install ubuntu-drivers-common nvidia-cuda-toolkit
sudo ubuntu-drivers autoinstall

#Then reboot your system for the driver to be properly loaded.
sudo reboot
```

Install and start deepstack (if using GPU):
```
# install nvidia container toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
# test and make sure the GPU is visible from a docker container
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# then install and run the deepstack container on port 5000
sudo docker pull deepquestai/deepstack:gpu-2021.09.1
# in the command below, replace `/home/local/oddspot/` with your oddspot directory
sudo docker run --gpus all -e VISION-DETECTION=True -v localstorage:/datastore -v /home/local/oddspot/custom_models:/modelstore/detection -p 5000:5000 -d --restart always deepquestai/deepstack:gpu
```

Install and start deepstack (if using CPU):
```
docker pull deepquestai/deepstack
# in the command below, replace `/home/local/oddspot/` with your oddspot directory
docker run -e VISION-DETECTION=True -v localstorage:/datastore -v /home/local/oddspot/custom_models:/modelstore/detection -p 80:5000 -d --restart always deepquestai/deepstack
```


## Sample config

Create a file `oddspot.ini` within the `oddspot` base directory containing the following:

```
[detection]
#min_notify_period: minimum object identification confidence level in order to notify
# we will notify if any identified object in the image has >= this level
# e.g. 0.2 = 20%, 0.7 = 70%
min_confidence=0.7

deepstack_api_port=5000

[notify]
pushover_user_key=<YOUR USER KEY HERE>
pushover_api_token=<YOUR API KEY HERE>

#min_notify_period: only send out a pushover notification for a given camera this often (in seconds)
min_notify_period=600

#notify_on_dataset_categories: yolov5 / COCO dataset labels to notify on
notify_on_dataset_categories=person,bicycle,car,motorcycle,bus,truck

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
Group=<USER GROUP>
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

