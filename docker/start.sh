#!/usr/bin/env bash
# TO RECONNECT WITH A WORKING AUTH SOCKET: docker exec -it -e SSH_AUTH_SOCK=$SSH_AUTH_SOCK container_ID /bin/bash
USER_LOWER=$(echo "$USER" | awk '{print tolower($0)}')

<<<<<<< HEAD
image_name=mae_stereo_v3
container_name=mae_stereo_v3.0_v2_${USER_LOWER}
=======
image_name=mae_stereo_v4
container_name=mae_stereo_v4.0_v1_${USER_LOWER}
>>>>>>> 809dac2454accbbc6e211dc4527985107ad491bd

# Default args.
devices=all
if [ $devices != 'all' ]
then
  echo "Using devices {$devices}"
  devices="'\"device=${devices}\"'"
fi

command=/bin/bash

unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    *)          machine="UNKNOWN:${unameOut}"
esac
echo "running on ${machine}"

if [ ${machine} = 'Mac' ]
then
  mkdir -p $HOME/Documents/mae_stereo_v2-data/data/
  volumes="-v $HOME/Documents/mae_stereo-data/data:/data"
  mount="--mount type=bind,source=${HOME},target=/home/${USER}"
  gpus=""
else
  volumes="-v /data:/data -v /image1:/image1"
  mount="--mount type=bind,source=/data,target=/home/${USER}/data \
         --mount type=bind,source=${HOME},target=/home/${USER}"
  gpus="--gpus ${devices}"
fi

if [ ! -z "${SSH_AUTH_SOCK}" ]; then
    vol_path=$(dirname $(dirname ${SSH_AUTH_SOCK}))
    ssh_agent="-v ${vol_path}:${vol_path} --env SSH_AUTH_SOCK=${SSH_AUTH_SOCK}"
    volumes=${volumes}" "${ssh_agent}
fi

eval "docker pull ${image_name}:latest"
eval "docker run -it \
           --ulimit nofile=1000000:1000000 \
           -u $(id -u):$(id -g) \
           -w /home/${USER} \
           --cap-add=SYS_PTRACE \
           --security-opt seccomp=unconfined \
           --ipc=host \
           -v /etc/passwd:/etc/passwd \
           -v /etc/group:/etc/group \
           ${volumes} \
           ${mount} \
           --env HOME=/home/${USER} \
           --network=host \
           ${gpus} \
           ${extra_args} \
           --name ${container_name} \
           --rm \
           ${image_name} ${command}"
