import timm

from src.utils.builder import build_trainer, build_config
from src.utils.utils import *
from src.utils.builder import *
from src.utils.fileio import *
from src.models.modules.image_encoder import *
from functools import partial 
from arguments import args
from time import time
from PIL import Image
import numpy as np
import torchvision.transforms as transforms 
import torch 
import torch.nn as nn
import timm 
from thop import profile

# --------------- configure initialisers -----------------
setup_imports()
    
parser= args.get_parser()
opts = parser.parse_args()

config = build_config(opts)

print('selected patch size: {}'.format(config.model_config.image_encoder.patch_size))

#model = MaskedImageAutoEncoderMSGGAN(config)
img_size = config.dataset_config.preprocess.vision_transforms.params.Resize.size
patch_size = config.model_config.image_encoder.patch_size
in_channels = config.model_config.image_encoder.in_channels
embed_dim = config.model_config.image_encoder.embed_dim
norm_layer_arg= config.model_config.norm_layer_arg
        
if patch_size[0]!=patch_size[1]:
    print('patch sizes are non-square, so we will download ViT form timm with weights but change the patch weights to randomly intialised ones')
else:
    print('patch size is square: {}, if it is 8 or 16, then ViT from timm will be downloaded as normal'.format(patch_size))

if norm_layer_arg=='partial':
    norm_layer = partial(nn.LayerNorm, eps=1e-6)
    print('using partial layer norm')
else:
    norm_layer = nn.LayerNorm

patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)

supervised_model = timm.create_model('vit_base_patch16_224', pretrained=True)

# Change patch embed weights if it is not square
if patch_size[0]!=patch_size[1]:
    supervised_model.patch_embed = patch_embed
save_dir = "/data/tiawarner/imagenet_weights"
if os.path.exists(save_dir)!=True:
    os.makedirs(save_dir)
    print("output save directory made: {}".format(save_dir))
else:
    print("save directory: {} Already Exists!".format(save_dir))

#save the weights
torch.save(supervised_model.state_dict(), '{}/vit_base_patch16_224.pth'.format(save_dir))
