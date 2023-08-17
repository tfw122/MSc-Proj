FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

ARG PYTORCH="1.12.1"
ARG CUDA="11.3"
ARG CUDNN="8"

# can use the latest version should you choose
ARG MMCV="1.3.14" 
ARG MMDET="2.18.0"

ENV DEBIAN_FRONTEND noninteractive
ENV TERM=xterm-256color
ENV FORCE_COLOR_PROMPT=yes
ENV HOME=/home

# set env var to locate where all the notebook templates are.
ENV VIRTUAL_ENV=/opt/venv

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    curl \
    ca-certificates \
    dirmngr \
    git \ 
    transmission-cli \
    aria2 \
    language-pack-ja \
    libgl1-mesa-dev \
    libpq-dev \
    transmission-daemon \
    transmission-cli \
    libturbojpeg \
    libopencv-dev \
    libfreetype6-dev \
    libturbojpeg0-dev \
    lsb-release \
    libfmt-dev \
    liblz4-dev \
    nlohmann-json-dev \
    openssh-server \
    poppler-utils \
    pkg-config \
    p7zip-full \
    p7zip-rar  \
    jq \
    g++ \
    cmake \
    python3-pip \
    python3-venv \
    software-properties-common \
    sudo \
    tmux \
    nano \
    htop \
    tzdata \
    unzip \
    vim \
    p7zip-full \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-regex-dev \
    libboost-system-dev \
    libboost-test-dev \
    libdouble-conversion-dev \
    libeigen3-dev \
    libflann-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libsqlite3-dev \
    libsuitesparse-dev \
    libatlas-base-dev \
    libfreeimage-dev \
    protobuf-compiler \
    libprotobuf-dev \
    libglew-dev \
    libglvnd-dev \
    libopencv-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev \
    libgtk2.0-dev \
    libceres-dev \
    npm \
    colmap \
    ffmpeg \
    libsm6 \
    libxext6 \
    ninja-build \
    libglib2.0-0 \
    libxrender-dev \
    xvfb \
    x11vnc \
    && rm -rf /var/lib/apt/lists/*


ENV VIRTUAL_ENV=/opt/venv

# environment, dependencies and entry points
ENV SPARK_HOME="/opt/spark"
ENV JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"
ENV PATH="$SPARK_HOME/bin:$PATH"
ENV VIRTUAL_ENV=/opt/venv
# upgrade: MMCV integration
RUN export FORCE_CUDA=1

WORKDIR /home

# Install Azure Copy for file transfer via CLI
RUN echo "Setup AzCopy.."
RUN wget https://aka.ms/downloadazcopy-v10-linux \
    && tar -xvf downloadazcopy-v10-linux \
    # Move AzCopy
    && sudo cp ./azcopy_linux_amd64_*/azcopy /usr/bin/ \
    && sudo chmod 755 /usr/bin/azcopy \
    # Clean the kitchen
    && rm -f downloadazcopy-v10-linux \
    && rm -rf ./azcopy_linux_amd64_*/
    
RUN echo "AzCopy setup completed!"

# to copy in azure;
# azcopy copy "file_name.exension" "container sas token" --recursive=true

# Install npm >>> via apt install
#RUN curl -sL https://deb.nodesource.com/setup_18.x | sudo -E bash - \
#  && apt-get -y install nodejs \
#  && rm -rf /var/lib/apt/lists/* \ 
#  && rm /etc/apt/sources.list.d/nodesource.list


# Avoid a shared jupyter config
ENV JUPYTER_CONFIG_DIR /jupyter/.jupyter
ENV JUPYTER_DATA_DIR /jupyter/.local/share/jupyter
ENV JUPYTER_RUNTIME_DIR /jupyter/.local/share/jupyter/runtime
RUN mkdir -p $JUPYTER_CONFIG_DIR \
  && mkdir -p $JUPYTER_DATA_DIR \
  && mkdir -p $JUPYTER_RUNTIME_DIR \
  && chmod -R 777 /jupyter

# Set Up Nexus repo
COPY pip.conf  $VIRTUAL_ENV/pip.conf

# Core Python libs
# 16/08: pip new resolver is causing backtracking issues
# reverting to old resolver until we have a fix
RUN python3 -m pip install virtualenv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN chmod -R 777 $VIRTUAL_ENV

COPY ./notebook_requirements.txt /tmp/requirements.txt
RUN pip3 install --upgrade setuptools wheel
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt
RUN python3 -m pip install --upgrade --no-cache-dir \
  cffi \
  cython \
  jupyterlab \
  ipywidgets \
  jupyter_contrib_nbextensions \
  jupyterlab-git \
  jupytext \
  nbconvert \
  notebook \
  pip \
  setuptools

# Scientist stack
COPY requirements.txt /opt/requirements.txt
COPY bash.bashrc /etc/bash.bashrc
RUN python3 -m pip install  --upgrade --no-cache-dir setuptools wheel ipython
RUN python3 -m pip install --use-deprecated=legacy-resolver --upgrade --no-cache-dir -r /opt/requirements.txt


# Installing consistent depth modules:
RUN git clone https://github.com/facebookresearch/consistent_depth.git && cd consistent_depth \
    && git submodule update --init --recursive \
    && bash ./scripts/install.sh \
    && bash ./scripts/install_colmap_ubuntu.sh \
    && bash ./scripts/download_model.sh

# upgrade: Detectron2 integration
RUN git clone https://github.com/facebookresearch/detectron2.git \
    && python3 -m pip install -e detectron2

# upgrade: MMCV integration
ARG PYTORCH
ARG CUDA
RUN ["/bin/bash", "-c", "pip install mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]
RUN python3 -m pip install --upgrade --no-cache-dir mmdet==${MMDET}

# upgrade: MMSegmentation integration
RUN git clone https://github.com/open-mmlab/mmsegmentation.git \
    && cd mmsegmentation \
    && python3 -m pip install -v -e .

ARG USER_NAME
ARG USER_ID

####### CREATE A USER #########
RUN adduser --disabled-password --gecos '' --uid ${USER_ID} --force-badname ${USER_NAME} \
    && usermod -aG sudo ${USER_NAME}

RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> \
    /etc/sudoers

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/:/usr/lib/
ENV PATH=$PATH:/usr/local/cuda/bin/
ENV PYTHONPATH=.
RUN mkdir -p -m 777 /data/azure_cache/

CMD ["/bin/bash"]
