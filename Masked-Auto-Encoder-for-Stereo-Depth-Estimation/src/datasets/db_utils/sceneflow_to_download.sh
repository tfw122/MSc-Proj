#!/bin/bash

mkdir ./scene_flow
cd scene_flow

# CLEANPASS
# download the raw_data i.e. images;
wget --no-check-certificate http://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/raw_data/flyingthings3d__frames_cleanpass.tar
wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Driving/raw_data/driving__frames_cleanpass.tar
wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Monkaa/raw_data/monkaa__frames_cleanpass.tar

# FINALPASS
# download the raw_data i.e. images;
wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/raw_data/flyingthings3d__frames_finalpass.tar
wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Driving/raw_data/driving__frames_finalpass.tar
wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Monkaa/raw_data/monkaa__frames_finalpass.tar

# download the derived_data i.e. disparity / optical flow maps;
wget --no-check-certificate http://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/derived_data/flyingthings3d__disparity.tar.bz2
wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Driving/derived_data/driving__disparity.tar.bz2
wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Monkaa/derived_data/monkaa__disparity.tar.bz2

tar xvf flyingthings3d__frames_cleanpass.tar
tar xvf driving__frames_cleanpass.tar
tar xvf monkaa__frames_cleanpass.tar
tar xvf flyingthings3d__disparity.tar.bz2
tar xvf monkaa__disparity.tar.bz2