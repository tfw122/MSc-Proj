
#from src.utils.builder import build_trainer, build_config
from src.utils.utils import *
from src.utils.builder import *
from src.utils.fileio import *
from arguments import args
from time import time
from PIL import Image
import numpy as np
import random 
import pandas as pd 

import torch 
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image
from src.datasets.transforms.vision_transforms_utils import UnNormalise

# --------------- Import Dataset Builder function -----------------
from src.datasets.dataset_zoo.stereo_vision_mim.stereo_datasets_mim.cityscapes_dataset import *
from src.datasets.dataset_zoo.stereo_vision_mim.stereo_datasets_mim.kitti_dataset import *
from src.datasets.dataset_zoo.stereo_vision_mim.stereo_datasets_mim.falling_things_dataset import *
from src.datasets.dataset_zoo.stereo_vision_mim.stereo_datasets_mim.sceneflow_dataset import *
from src.datasets.dataset_zoo.stereo_vision_mim.stereo_datasets_mim.sintel_stereo_dataset import *
from src.datasets.dataset_zoo.stereo_vision_mim.stereo_datasets_mim.tartan_air_dataset import *

from src.datasets.dataset_zoo.stereo_vision_downstream.stereo_datasets_downstream.kitti_dataset import *
from src.datasets.dataset_zoo.stereo_vision_downstream.stereo_datasets_downstream.falling_things_dataset import *
from src.datasets.dataset_zoo.stereo_vision_downstream.stereo_datasets_downstream.sceneflow_dataset import *
from src.datasets.dataset_zoo.stereo_vision_downstream.stereo_datasets_downstream.sintel_stereo_dataset import *
from src.datasets.dataset_zoo.stereo_vision_downstream.stereo_datasets_downstream.tartan_air_dataset import *
from src.datasets.dataset_zoo.stereo_vision_downstream.stereo_datasets_downstream.eth3d_dataset import *
from src.datasets.dataset_zoo.stereo_vision_downstream.stereo_datasets_downstream.middlebury_dataset import *

global LOAD_IMG
global TESTING

LOAD_IMG= "dataloader"
TESTING = "downstream"
print("end")


def get_transforms(config, split):
    transforms_name = config.dataset_config.preprocess.name
    data_transform_cls = registry.get_preprocessor_class(transforms_name)
    data_transforms_obj = data_transform_cls(config, split)
    return data_transforms_obj

def ToArray(img_t):
    img = img_t.detach().to("cpu").numpy()
    img = np.transpose(img, (1, 2, 0))

    return img

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    if TESTING=='downstream':
        plt.imshow(image.numpy().astype(np.uint8)) # .numpy().astype(np.uint8)
    else:
        plt.imshow(image)
    plt.title(title, fontsize=16)
    plt.axis('off')


def augmentation_parameters(config):
    dataset_config = config.dataset_config
    preprocess_config = dataset_config.preprocess.vision_transforms.params
    aug_params = {'crop_size': preprocess_config.Resize.size, 'min_scale': preprocess_config.spatial_scale[0], 'max_scale': preprocess_config.spatial_scale[1], 'do_flip': preprocess_config.do_flip, 'yjitter': not preprocess_config.noyjitter}
    
    if hasattr(preprocess_config, "saturation_range") and preprocess_config.saturation_range is not None:
        aug_params["saturation_range"] = tuple(preprocess_config.saturation_range)
    
    if hasattr(preprocess_config, "img_gamma") and preprocess_config.img_gamma is not None:
        aug_params["gamma"] = preprocess_config.img_gamma
    
    if hasattr(preprocess_config, "do_flip") and preprocess_config.do_flip is not None:
        aug_params["do_flip"] = preprocess_config.do_flip
    return aug_params


def run_one_image(left_img, right_img, model, mask_ratio=None):
    # make it a batch-like
    left_img = left_img.unsqueeze(dim=0)
    right_img = right_img.unsqueeze(dim=0)
    #x = torch.einsum('nhwc->nchw', x)

    # run MAE
    y, mask = model(left_img, right_img, mask_ratio) #, mask_ratio=0.75)
    
    left_recon= y[0]
    right_recon = y[1]
    left_mask = mask[0]
    right_mask = mask[1]
    
    left_y = model.unpatchify(left_recon)
    right_y = model.unpatchify(right_recon)
    print(left_y.max(), left_y.min(), right_y.max(), right_y.min())
    left_y = torch.einsum('nchw->nhwc', left_y).detach().cpu()
    right_y = torch.einsum('nchw->nhwc', right_y).detach().cpu()

    # visualize the mask
    left_mask = left_mask.detach()
    left_mask = left_mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]*model.patch_embed.patch_size[1]*3)  # (N, H*W, p*p*3)
    left_mask = model.unpatchify(left_mask)  # 1 is removing, 0 is keeping
    left_mask = torch.einsum('nchw->nhwc', left_mask).detach().cpu()
    
    right_mask = right_mask.detach()
    right_mask = right_mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]*model.patch_embed.patch_size[1]*3)  # (N, H*W, p*p*3)
    right_mask = model.unpatchify(right_mask)  # 1 is removing, 0 is keeping
    right_mask = torch.einsum('nchw->nhwc', right_mask).detach().cpu()
    
    left_img = torch.einsum('nchw->nhwc', left_img)
    right_img = torch.einsum('nchw->nhwc', right_img)

    print("middle")

    # masked image
    left_im_masked = left_img * (1 - left_mask)
    right_im_masked = right_img * (1 - right_mask)
    
    # MAE reconstruction pasted with visible patches
    left_im_paste = left_img * (1 - left_mask) + left_y * left_mask
    right_im_paste = right_img * (1 - right_mask) + right_y * right_mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 48]

    plt.subplot(1, 6, 1)
    show_image(left_img[0], "Left original")
    
    plt.subplot(1, 6, 2)
    show_image(right_img[0], "Right original")

    plt.subplot(1, 6, 3)
    show_image(left_im_masked[0], "Left masked")
    
    plt.subplot(1, 6, 4)
    show_image(right_im_masked[0], " Right masked")

    plt.subplot(1, 6, 5)
    show_image(left_y[0], "Left reconstruction")
    
    plt.subplot(1, 6, 6)
    show_image(right_y[0], "Right reconstruction")

    #plt.subplot(1, 8, 7)
    #show_image(left_im_paste[0], "reconstruction + visible")
    
    #plt.subplot(1, 8, 8)
    #show_image(right_im_paste[0], "reconstruction + visible")

    plt.show()
    print("end")

class Args:
    default_config_path= './configs/default.yaml'
    model_config_path='./configs/models/masked_image.yaml'
    if TESTING=='downstream':
        dataset_config_path='./configs/datasets/stereo_downstream.yaml'
    else:
        dataset_config_path='./configs/datasets/stereo_mim.yaml'
    user_config_path='./configs/user/sample.yaml'
    local_rank=None
    opts=None

args=Args()
setup_imports()

config = build_config(args)

fileio_client = FileIOClient(config)
train_transforms = get_transforms(config, 'train')
test_transforms = get_transforms(config, 'test')

# load model; dataset etc;
print(os.getcwd())
model = build_model(config, ckpt_path= '../data/tiawarner/downstream4/mae_stereo_mim_perceptual/230824-210638/train_outputs/best-model-epoch=018-val_loss=0.52.ckpt')

print('::::::: model loaded with ckpt weights :::::::')

idx=0




img_path_left = "../data/middlebury/testH/Bicycle2/im0.png"
    #"/data/stereo_data/middlebury/MiddEval3/testF/Classroom2/im0.png"
img_path_right= "../data/middlebury/testH/Bicycle2/im1.png"
    #"/data/stereo_data/middlebury/MiddEval3/testF/Classroom2/im1.png"

    # pre-process;
left_img = Image.open(img_path_left).convert('RGB')
left_img = left_img.resize((448,224))

right_img = Image.open(img_path_right).convert('RGB')
right_img = right_img.resize((448,224))

assert np.shape(left_img) == (224, 448, 3)
assert np.shape(right_img) == (224, 448, 3)

    # normalize by ImageNet mean and std
    #img = img - imagenet_mean
    #img = img / imagenet_std
totensor = transforms.ToTensor()

plt.rcParams['figure.figsize'] = [5, 5]
show_image(totensor(left_img).permute(1,2,0), 'left image')
show_image(totensor(right_img).permute(1,2,0), 'right image')


totensor = transforms.ToTensor()
left_img_t = totensor(left_img)
right_img_t = totensor(right_img)
run_one_image(left_img_t, right_img_t, model)

    
else:
    totensor = transforms.ToTensor()
    torch.manual_seed(2) # <<< random seed for random masking.
    left_img_t = totensor(left_img)
    right_img_t = totensor(right_img)
    run_one_image(left_img_t, right_img_t, model, 0.75)