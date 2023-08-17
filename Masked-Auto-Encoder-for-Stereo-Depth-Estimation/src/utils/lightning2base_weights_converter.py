import numpy as np
import torch
import torch.nn as nn
import os 
import argparse
from collections import OrderedDict

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--input_path', type=str, default=' /media/neuralmaster/9d5af100-a900-4e89-bab1-43c8b5025daf/neuromaster/MainImportant/MRES/tiaW/Masked-Auto-Encoder-for-Stereo-Depth-Estimation/data/tiawarner/geico_and_nugen_mae_perceptual/gieco_and_nugen_mae_training_v2/train_outputs/sample-epoch=069-val_loss=0.66.ckpt', help="the directory where your lightning weights you'd like to convert are stored")
    parse.add_argument('--output_path', type=str, default=' /media/neuralmaster/9d5af100-a900-4e89-bab1-43c8b5025daf/neuromaster/MainImportant/MRES/tiaW/Masked-Auto-Encoder-for-Stereo-Depth-Estimation/data/tiawarner/weights/converted/', help="the directory where you'd like to store your new converted weights")
    parse.add_argument('--output_filename', type=str, default='COCO_weights.pth', help="the name of the new weight file")

    args = parse.parse_args()
    return args 

def load_pre_text_pretrained_weights(pretrained_mae_path):
    # load pretrained weights (from local);
    pretrained_weights = torch.load(pretrained_mae_path, map_location='cpu')
    pretrained_weights = pretrained_weights['state_dict']
    # these weights will have encoder attached in front of the dict keys.
    # we will clean this up;
    new_pretrained_weights_dict = OrderedDict()
    for k, v in pretrained_weights.items():
        name = k.replace('encoder.', '') # remove `encoder.` k[8:]
        #name = 'vit_model.'+ name # add `vit_model.` to make it identical to current model.
        new_pretrained_weights_dict[name] = v
    # 1. filter out unnecessary keys
    
    return new_pretrained_weights_dict


if __name__=='__main__':
    args = get_args()
    lightning_dict= args.input_path #'/data/samyakhtukra/geico_and_nugen_mae_perceptual/gieco_and_nugen_mae_training_v2/train_outputs'
    
    print('converting checkpoint Lightning weights: {}'.format(lightning_dict))

    new_dict = load_pre_text_pretrained_weights(lightning_dict)

    torch.save(new_dict,'{}/{}'.format(args.output_path, args.output_filename))
    
    print('new weights saved to: {}/COCO_VERSION_{}'.format(args.output_path, args.output_filename))
