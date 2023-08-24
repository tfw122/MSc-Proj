#!/bin/bash
mkdir /data/tiawarner/pretrained_raft/ -p
cd /data/tiawarner/pretrained_raft/
wget https://www.dropbox.com/s/q4312z8g5znhhkp/models.zip
unzip models.zip
rm models.zip -f