#from src.utils.builder import build_trainer, build_config
from src.utils.utils import *
from src.utils.builder import *
from src.utils.fileio import *
from arguments import args
from time import time
from PIL import Image
import numpy as np
import torchvision.transforms as transforms 
import torch 
import torch.nn as nn
import timm 
from thop import profile
import argparse

from src.models.modules.image_encoder import *
from functools import partial 

# --------------- import the model ----------------
from src.models.model_zoo.masked_vision_model import *
from src.models.modules.masked_vision_layers import *
from src.models.model_zoo.vector_quantized_mae import VQMaskedImageAutoEncoder
from src.models.modules.raft_modules.core.raft_stereo import *
from src.models.model_zoo.masked_vision_stereo_downstream import *

# --------------- Helper function initialisers -----------------

def raft_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/testH/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/testH/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()
    return args

# --------------- configure initialisers -----------------
setup_imports()
parser= args.get_parser()
opts, unknown = parser.parse_known_args()

args_raft_stereo = raft_args()

config = build_config(opts)

# setup aws env variables:
s3_client = FileIOClient(config)

# check correct config loaded;
print(config.model_config.name)
USE_DATASET=False

# --------------- Initialise the model ----------------
#model = MaskedImageAutoEncoderMSGGAN(config)
img_size = config.dataset_config.preprocess.vision_transforms.params.Resize.size
patch_size = config.model_config.image_encoder.patch_size
in_channels = config.model_config.image_encoder.in_channels
embed_dim = config.model_config.image_encoder.embed_dim
norm_layer_arg= config.model_config.norm_layer_arg
        
if norm_layer_arg=='partial':
    norm_layer = partial(nn.LayerNorm, eps=1e-6)
    print('using partial layer norm')
else:
    norm_layer = nn.LayerNorm

patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)

#encoder = MSGMAEEncoder(config, patch_embed, norm_layer)
#decoder = MSGMAEDecoder(config, patch_embed, norm_layer)
model = RAFTStereo(args_raft_stereo)
vit_raft = StereoVITEncoderDownStream(config)
vit_raft.eval();
model.eval();

print("raft params:", count_parameters(model))
print("mae-raft params:", count_parameters(vit_raft))
print("difference: ", count_parameters(vit_raft) - count_parameters(model))
# create random input tensor:
left_image = torch.randn((8,3,224,448))
right_image = torch.randn((8,3,224,448))

# ----------------- model inference ----------------
output = model(left_image, right_image, iters=config.model_config.raft_decoder.train_iters)

output_x = vit_raft(left_image, right_image, iters=config.model_config.raft_decoder.train_iters)

print("------------- FULL MODEL FLOPS -------------")
macs, params = profile(model, inputs=(left_image, right_image, config.model_config.raft_decoder.train_iters))
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

print("------------- FULL MODEL MAE - RAFT FLOPS -------------")
macs, params = profile(vit_raft, inputs=(left_image, right_image, config.model_config.raft_decoder.train_iters))
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
