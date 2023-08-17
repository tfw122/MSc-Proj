import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from pathlib import Path
import random
import timm
import timm.optim.optim_factory as optim_factory
from functools import partial 
import os 
from collections import OrderedDict


import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from src.models.modules.vqgan_modules.main import instantiate_from_config

#from src.models.modules.vqgan_modules.taming.modules.diffusionmodules.model import Encoder, Decoder
from src.models.modules.vqgan_modules.taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from src.models.modules.vqgan_modules.taming.modules.vqvae.quantize import GumbelQuantize, EMAVectorQuantizer

from src.models.modules.pos_embeds import *
from src.models.base_model import BaseModel
from src.models.modules.image_encoder import *
from src.models.modules.masked_vision_layers import *
from src.common.registry import registry
from src.models.modules.layer_utils import *
from src.losses.image_reconstruction import MaskedImageLoss, scale_pyramid
from src.utils.utils import load_yaml
from src.datasets.transforms.vision_transforms_utils import UnNormalise
from src.common.constants import IMAGE_COLOR_MEAN, IMAGE_COLOR_STD



@registry.register_model("vq_stereo_masked_autoencoder")
class VQStereoMaskedImageAutoEncoder(BaseModel):
    def __init__(self, config):
        super().__init__()
        print('for this work well, make sure inputs are not normalised!')
        self.config = config
        self.model_config = self.config.model_config
        self.vq_config = self.model_config.vector_quantizer
        self.ddconfig = self.vq_config.ddconfig
        self.lossconfig = self.vq_config.lossconfig
        self.dataset_config =  self.config.dataset_config
        self.user_config = self.config.user_config
        self.image_out_dir= '{}/{}/mae_out_test_{}_{}_{}'.format(self.config.user_config.save_root_dir, self.config.user_config.username_prefix, self.dataset_config.dataset_name, self.model_config.loss_type, self.user_config.experiment_name)
        if os.path.exists(self.image_out_dir)!=True:
            os.makedirs(self.image_out_dir)
        # patch embed args;
        self.mask_ratio = self.model_config.mask_ratio
        self.finetune_imagenet= self.model_config.finetune_imagenet
        self.num_samples_to_visualise = self.model_config.num_samples_to_visualise
        self.frequency_to_visualise = self.model_config.frequency_to_visualise

        img_size = self.dataset_config.preprocess.vision_transforms.params.Resize.size
        self.patch_size = self.model_config.image_encoder.patch_size
        in_channels = self.model_config.image_encoder.in_channels
        embed_dim = self.model_config.image_encoder.embed_dim
        self.norm_layer_arg= self.model_config.norm_layer_arg
        
        # VQGAN args;

        if self.norm_layer_arg=='partial':
            self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
            print('using partial layer norm')
        else:
            self.norm_layer = nn.LayerNorm
        
        self.patch_embed = PatchEmbed(img_size, self.patch_size, in_channels, embed_dim)
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.encoder = MAEEncoder(config, self.patch_embed, self.norm_layer)
        # --------------------------------------------------------------------------
        # VQGAN Quantizer specifics
        self.quantize = VectorQuantizer(self.vq_config.n_embed, self.vq_config.embed_dim, beta=self.vq_config.beta,
                                        remap=self.vq_config.remap, sane_index_shape= self.vq_config.sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(self.ddconfig["z_channels"], self.vq_config.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(self.vq_config.embed_dim, self.ddconfig["z_channels"], 1)

        # MAE decoder specifics
        self.decoder = MAEDecoder(config, self.patch_embed, self.norm_layer)
        # --------------------------------------------------------------------------

        # --------- build loss ---------
        self.loss_fnc = instantiate_from_config(self.lossconfig)
        #self.loss_fnc = MaskedImageLoss(config, self.patch_embed)
        
        if self.model_config.normalisation_params=='imagenet':
            self.unnormalise = UnNormalise(IMAGE_COLOR_MEAN, IMAGE_COLOR_STD)
        else:
            raise Exception("the following type of normalisation has not been implemented: {}".format(self.model_config.normalisation_params))

        if self.finetune_imagenet!=None:
            self.load_imagenet_weights()
            print('og imagenet weights loaded from: {} \n to commence finetuning'.format(self.finetune_imagenet))

    def load_imagenet_weights(self):
        # load the model dictionary
        # NOTE: ONLY encoder weights should be loaded; decoder has to be trained from scratch for the specific data;
        # otherwise no point in doing MAE since imagenet distribution can also reconstruct different image types.
        #pretrained_dict= torch.load(self.finetune_imagenet)
        pretrained_dict= torch.load(self.finetune_imagenet)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']

        model_dict = self.encoder.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.encoder.load_state_dict(model_dict)
        # if patch embed value != pretrained dict; throw an error; to ensure patchembed is correct.
        
    def forward(self, x_left, x_right):
        # <<<<<<<<<<<<<<<< LEFT IMAGE INFERENCE >>>>>>>>>>>>>>>>>>>
        ################## ENCODE ##################
        # run encoder;
        l_latent, l_mask, l_ids_restore = self.encoder(x_left, self.mask_ratio)
        # quantize;
        l_h = self.quant_conv(l_latent.unsqueeze(1))
        l_quant, l_emb_loss, l_info = self.quantize(l_h)
        
        ################## DECODE ##################
        # run decoder;
        l_quant = self.post_quant_conv(l_quant).squeeze(1)
        l_pred = self.decoder(l_quant, l_ids_restore)  # instead of latent; you take in the quant [N, L, p*p*3]

        # <<<<<<<<<<<<<<<< RIGHT IMAGE INFERENCE >>>>>>>>>>>>>>>>>>>
        ################## ENCODE ##################
        # run encoder;
        r_latent, r_mask, r_ids_restore = self.encoder(x_right, self.mask_ratio)
        # quantize;
        r_h = self.quant_conv(r_latent.unsqueeze(1))
        r_quant, r_emb_loss, r_info = self.quantize(r_h)
        
        ################## DECODE ##################
        # run decoder;
        r_quant = self.post_quant_conv(r_quant).squeeze(1)
        r_pred = self.decoder(r_quant, r_ids_restore)  # instead of latent; you take in the quant [N, L, p*p*3]

        return (l_pred, r_pred), (l_mask, r_mask), (l_quant, r_quant), (l_emb_loss, r_emb_loss)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        x_left, x_right = batch['left_image'], batch['right_image']

        out, mask, _, qloss= self(x_left, x_right)
        
        # convert out to reconstructed output;
        xrec_left = self.unpatchify(out[0])
        xrec_right = self.unpatchify(out[1])
        qloss_left = qloss[0]
        qloss_right = qloss[1]

        # log images;
        if self.global_step % self.frequency_to_visualise ==0:
            preds_left = out[0].clone().detach() # left image
            preds_right = out[1].clone().detach() # right image

            mask_left = mask[0].clone().detach() # left mask
            mask_right = mask[1].clone().detach() # right mask
            
            # [rand_batch_id,:,:].unsqueeze(0)
            orig_img_left, masked_img_left, recon_left, _= self.visualise_sample(preds_left[0:2,:,:], 
                                                                                 mask_left[0:2,:], 
                                                                                 x_left[0:2,:,:,:])

            # _, _, _, recon_with_visible_right
            orig_img_right, masked_img_right, recon_right, _= self.visualise_sample(preds_right[0:2,:,:], 
                                                                                    mask_right[0:2,:], 
                                                                                    x_right[0:2,:,:,:])
            # deifne grid;
            grid = make_grid(
                torch.cat((orig_img_left.permute(0,3,1,2), orig_img_right.permute(0,3,1,2), 
                           masked_img_left.permute(0,3,1,2), masked_img_right.permute(0,3,1,2),
                           recon_left.permute(0,3,1,2), recon_right.permute(0,3,1,2)), dim=0))

            #self.logger.experiment.add_image('train_images', grid, batch_idx, self.global_step)
            
            save_image(grid, '{}/{:08d}.png'.format(self.image_out_dir, self.global_step))
        
        if optimizer_idx == 0:
            # autoencode
            aeloss_l, log_dict_ae_l = self.loss_fnc(qloss_left, x_left, xrec_left, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            aeloss_r, log_dict_ae_r = self.loss_fnc(qloss_right, x_right, xrec_right, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            aeloss = aeloss_l + aeloss_r
            log_dict_ae = log_dict_ae_l + log_dict_ae_r
            
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss_l, log_dict_disc_l = self.loss_fnc(qloss_left, x_left, xrec_left, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            
            discloss_r, log_dict_disc_r = self.loss_fnc(qloss_right, x_right, xrec_right, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            
            discloss = discloss_l + discloss_r
            log_dict_disc = log_dict_disc_l + log_dict_disc_r

            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x_left, x_right = batch['left_image'], batch['right_image']

        out, mask, _, qloss= self(x_left, x_right)
        
        # convert out to reconstructed output;
        xrec_left = self.unpatchify(out[0])
        xrec_right = self.unpatchify(out[1])
        qloss_left = qloss[0]
        qloss_right = qloss[1]
        
        aeloss_l, log_dict_ae_l = self.loss_fnc(qloss_left, x_left, xrec_left, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        aeloss_r, log_dict_ae_r = self.loss_fnc(qloss_right, x_right, xrec_right, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        aeloss = aeloss_l + aeloss_r
        log_dict_ae = log_dict_ae_l + log_dict_ae_r

        discloss_l, log_dict_disc_l = self.loss_fnc(qloss_left, x_left, xrec_left, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        discloss_r, log_dict_disc_r = self.loss_fnc(qloss_right, x_right, xrec_right, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        discloss = discloss_l + discloss_r
        log_dict_disc = log_dict_disc_l + log_dict_disc_r
        
        rec_loss = log_dict_ae["val/rec_loss"]
        
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        #self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

        return self.log_dict

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    
    def configure_optimizers(self):
        """
        Configure and load optimizers here.
        """
        weight_decay=0.05
        blr= 4.5e-6
        min_lr = 0.
        warmup_epochs=20
        betas= (0.9, 0.95)

        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=blr, betas=(0.5, 0.9))
        
        opt_disc = torch.optim.Adam(self.loss_fnc.discriminator.parameters(),
                                    lr=blr, betas=(0.5, 0.9))
        
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9) #build_scheduler(optimizer, self.config)
        return [opt_ae, opt_disc], []

    # --------------- helper functions ----------------
    #def on_train_epoch_start(self):
    #    if self.current_epoch==0:
    #        sample_input= torch.randn((8,3,10,224,224))
    #        self.logger.experiment.add_graph(MaskedImageAutoEncoder(self.config),sample_input)

    def get_last_layer(self):
        return self.decoder.decoder_blocks[7].mlp.fc2.weight

    def visualise_sample(self, pred, mask, img):
        y = self.unpatchify(pred)
        y = torch.einsum('nchw->nhwc', y).detach() #.cpu()

        # visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_embed.patch_size[0]*self.patch_embed.patch_size[1]*3)  # (N, H*W, p*p*3)
        mask = self.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach() #.cpu()
        
        x = torch.einsum('nchw->nhwc', img)

        # masked image
        im_masked = x * (1 - mask)

        # model reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask

        # return (original image, masked image, model reconstruction, fused; reconstruction + visible pixels)
        return x[0], im_masked[0], y[0], im_paste[0]

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        #p = self.patch_embed.patch_size[0]
        ph = self.patch_embed.patch_size[0]
        pw = self.patch_embed.patch_size[1]

        #assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // ph
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, ph, w, pw))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, ph*pw * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        #p = self.patch_embed.patch_size[0]
        ph = self.patch_embed.patch_size[0]
        pw = self.patch_embed.patch_size[1]
        
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, ph, pw, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3,h * ph, w * pw))
        return imgs
