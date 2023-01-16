# Download base image ubuntu 22.04
FROM ubuntu:22.04

LABEL com.centurylinklabs.watchtower.enable="false"

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive

# Update Ubuntu Software repository
RUN apt update
RUN apt upgrade -y

# Install dependencies
RUN apt-get -y install git python3 python3-setuptools python3-pip 
# install python opencv system dependencies (see https://stackoverflow.com/a/63377623)
RUN apt-get -y install ffmpeg libsm6 libxext6 
# install python package deps
RUN pip3 install -U git+https://github.com/almir1904/python-pushover.git#egg=python-pushover aiosmtpd deepstack-sdk cmapy opencv-python-headless

# Clean up
RUN rm -rf /var/lib/apt/lists/*
RUN apt clean

# Add default user (with UID 35501 and GID 35501)
RUN groupadd -r oddspot -g 35501 && useradd --no-log-init -r -g oddspot -u 35501 oddspot

# Copy over sources
RUN mkdir -p /home/oddspot/state /home/oddspot/conf
COPY oddspot.py util.py /home/oddspot

#adjust permissions
RUN chown -R oddspot:oddspot /home/oddspot

# Volume configuration
VOLUME ["/home/oddspot/state"]

#Run program as our user
USER oddspot:oddspot
CMD ["/home/oddspot/oddspot.py"]

# Expose Port for the Application 
EXPOSE 10025
