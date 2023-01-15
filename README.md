# oddspot
Daemon that receives emails from cameras (with motion detection images attached) and uses [Deepstack](https://deepstack.cc/) to identify and tag the presence of people, vehicles, et cetra. If items of interest are located, an alert is sent out via [Pushover](https://pushover.net/).

## Installation

`oddspot` installs as a Docker Compose stack.
These instructions are for a **stock Ubuntu 22.04 LTS system**.

Install base dependencies:
```
sudo apt-get -y install sendemail curl
curl -fsSL get.docker.com -o get-docker.sh && sudo sh get-docker.sh
#add your user to the docker group
sudo adduser $USER docker
#either log out and back in, or run this next command for the group changes to take effect...
su - $USER
```

If you have a CUDA-capable NVIDIA GPU installed in your system, `oddspot` will work with it automatically, as long as the drivers are installed. 
```
sudo apt-get -y install ubuntu-drivers-common nvidia-cuda-toolkit
sudo ubuntu-drivers autoinstall

#Then reboot your system for the driver to be properly loaded.
sudo reboot
```

Note that if you run into an "unmet dependencies" error during running of the `ubuntu-drivers autoinstall` command, try the following:
```
sudo apt-get remove --purge nvidia-* -y
sudo apt autoremove
sudo ubuntu-drivers autoinstall
```

If using a GPU -- Install Nvidia container toolkit as well:
```
# install nvidia container toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
# test and make sure the GPU is visible from a docker container
docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```

## Sample config

Copy the `conf/oddspot.ini.dist` file to `conf/oddspot.ini` and customize it to your needs.
Once created, set the proper permissions:
```
sudo chown ${USER}:35501 conf/oddspot.ini
sudo chmod 640 conf/oddspot.ini
```

## Testing

If you have a GPU:
```
docker compose build
docker compose up
```

If using a CPU:
```
docker compose build
docker compose -f docker-compose-cpu.yml up
```

Then, copy a camera capture image to the system, and run the following command in another terminal window to send a test email that `oddspot` should pick up and process:

`sendemail -f test@test.com -t oddspot@localhost -s localhost:10025 -u 'testing 123' -m 'test' -a <TESTIMAGE>.jpg`

Tail `/home/<USER>/oddspot/logs/oddspot.log` to see if processing was successful or not.

## Usage

Just launch `docker compose up -d` (or `docker compose -f docker-compose-cpu.yml up -d` for CPU-only installs). The service will properly launch after a reboot.

Modify your camera setup (via its embedded web configuration page) to send an email to `oddspot@<YOURMACHINE>:<YOURPORT>` on motion detection events. Make sure it attaches a snapshot of the event in jpeg or png format (and _not_ a video). Also, to avoid flooding the script, you should probably have it so that it will only send repeat emails every minute or more. The `oddspot` script allows you to further limit continuous emails on a per camera basis via the `min_notify_period` config setting, but it should get raw emails from the cameras on motion events more frequently than that, to maximize alerting accuracy).

