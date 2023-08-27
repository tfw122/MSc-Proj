#!/usr/bin/env bash

# Some usernames have uppercase in them. Docker build doesn't like that.
USER_LOWER=$(echo "${USER,,}")

set -e


docker build --no-cache -f Dockerfile -t mae_stereo_v4 \
             --build-arg USER_NAME=${USER} \
             --build-arg USER_ID=$(id -u ${USER}) .
