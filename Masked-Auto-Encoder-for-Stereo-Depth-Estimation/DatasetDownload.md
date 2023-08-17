# Downloading Datasets for this repository 💾
The datasets are huge! make sure you have atleast **4 TB** space in your local / cloud storage to allow both downloading and uncompressing the files.

## Kitti dataset 

You can download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```shell
wget -i ~/Masked-Auto-Encoder-for-Stereo-Depth-Estimation/src/datasets/db_utils/kitti_archives_to_download.txt -P <$ROOT_DIR>/kitti_data/
```
Where `<$ROOT_DIR>` is where you'd like to store the output kitti data. Then unzip with

```shell
cd <$ROOT_DIR>/kitti_data
unzip "*.zip"
cd ..
```

**Warning:** it weighs about **175GB**, so make sure you have enough space to unzip too!

## Cityscapes
Just run the following commands to install cityscales left and right image pair dataset. **Before runnning the following, make sure to add cityscapes credentials in line 1 specifically for `$USERNAME` and `$PASSWORD` (remove the `$` ofcourse).
```shell
mkdir <$ROOT_DIR>/cityscapes
cd <$ROOT_DIR>/cityscapes
sh ~/Masked-Auto-Encoder-for-Stereo-Depth-Estimation/src/datasets/db_utils/cityscapes_to_download.sh
```

## Sceneflow
Run the following to install all 3 sub-datasets of sceneflow, i.e. FlyingThings3D, Monkaa and Driving.

```shell
mkdir <$ROOT_DIR>/sceneflow
cd <$ROOT_DIR>/sceneflow
sh ~/Masked-Auto-Encoder-for-Stereo-Depth-Estimation/src/datasets/db_utils/cityscapes_to_download.sh
```

## Tartan Air
For tartan air, you need to clone the following repository: [Tartan-Air-Tools](https://github.com/castacks/tartanair_tools). and simply follow the instructions form their ReadMe to download the data you deem fit. In our case, we did the following:

```shell
git clone https://github.com/castacks/tartanair_tools
cd tartanair_tools
python download_training.py --output-dir $OUTPUT_DIR --rgb --depth --flow
```

## Falling Things
Downloading this dataset is a bit tideous. It requires you to use `gdown`, as the dataset is provided as a googledrive link, which can be found on their [project page](https://research.nvidia.com/publication/2018-06_falling-things-synthetic-dataset-3d-object-detection-and-pose-estimation)

To download, use the following command in the terminal:
```shell
gdown --id 1y4h9T6D9rf6dAmsRwEtfzJdcghCnI_01 --output fat.zip
```
## ETH3D and Middlebury
Simply do the following:

```shell
sh ~/Masked-Auto-Encoder-for-Stereo-Depth-Estimation/src/datasets/db_utils/download_eth3d_and_middlebury.sh
```

## Sintel Stereo