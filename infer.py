from __future__ import annotations

import math
import random
import sys
from argparse import ArgumentParser
import os
import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
import shutil
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from blip_models.blip import blip_decoder
import json
import os



sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config

import json


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale,seg_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=4)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=4)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0],uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat1": [torch.cat([cond["c_concat1"][0], cond["c_concat1"][0], uncond["c_concat1"][0], uncond["c_concat1"][0]])],
            "c_concat2": [torch.cat([cond["c_concat2"][0], cond["c_concat2"][0],cond["c_concat2"][0], uncond["c_concat2"][0]])],
        }
        out_cond, out_img_cond, out_seg_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(4)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_seg_cond)+seg_cfg_scale * (out_seg_cond - out_uncond)



def get_text_for_image(image_filename, json_file):
    with open(json_file, 'r', encoding='utf-8') as infile:
        image_text_data = json.load(infile)
    
    if image_filename in image_text_data:
        return image_text_data[image_filename]
    else:
        return None

def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

def load_demo_image(image_size,device,img_url):
    
    raw_image = Image.open(img_url).convert('RGB')   

    w,h = raw_image.size
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0)   
    return image

def main():  
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--input", default= "C:\\Users\\Seo\\Desktop\\Gits\\ProjectPractice\\DiffV2IR-main\\input_folder\\input", type=str)
    parser.add_argument("--output", default= "C:\\Users\\Seo\\Desktop\\Gits\\ProjectPractice\\DiffV2IR-main\\output_folder", type=str)
    parser.add_argument("--edit", default="turn the RGB image into the infrared one",type=str)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--cfg-seg", default=1.5, type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    #os.makedirs('/home/jovyan/.cache/torch/hub/checkpoints/')
    #shutil.copy("checkpoint_liberty_with_aug.pth","/home/jovyan/.cache/torch/hub/checkpoints/")
    

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])
    pre = 'C:\\Users\\Seo\\Desktop\\Gits\\ProjectPractice\\DiffV2IR-main\\blip_models\\model__base_caption.pth'

    blip_model = blip_decoder(pretrained=pre, image_size=384, vit='base')
    blip_model.eval()

    seed = random.randint(0, 100000) if args.seed is None else args.seed
    #print(args.input)
    #print(os.walk(args.input))
    for root, dirs, files in os.walk(args.input):
        for file in files:
            image = load_demo_image(image_size=384, device='cuda',img_url=os.path.join(root,file))
            with torch.no_grad():
                caption = blip_model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) 
            args.edit = "turn the visible image of "+caption[0]+" into infrared"
            input_image = Image.open(os.path.join(args.input,file)).convert("RGB")
            input_seg = Image.open(os.path.join(args.input+"_seg",file.split(".")[0]+".png")).convert("RGB")
            width, height = input_image.size
            factor = args.resolution / max(width, height)
            factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
            width = int((width * factor) // 64) * 64
            height = int((height * factor) // 64) * 64
            input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)
            input_seg = ImageOps.fit(input_seg, (width, height), method=Image.Resampling.LANCZOS)

            if args.edit == "":
                input_image.save(os.path.join(args.output,file))
                return

            with torch.no_grad(), autocast("cuda"), model.ema_scope():
                cond = {}
                cond["c_crossattn"] = [model.get_learned_conditioning([args.edit])]
                input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
                input_seg = 2 * torch.tensor(np.array(input_seg)).float() / 255 - 1
                input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
                input_seg = rearrange(input_seg, "h w c -> 1 c h w").to(model.device)
                cond["c_concat1"] = [model.encode_first_stage(input_image).mode()]
                cond["c_concat2"] = [model.encode_first_stage(input_seg).mode()]

                uncond = {}
                uncond["c_crossattn"] = [null_token]
                uncond["c_concat1"] = [torch.zeros_like(cond["c_concat1"][0])]
                uncond["c_concat2"] = [torch.zeros_like(cond["c_concat2"][0])]

                sigmas = model_wrap.get_sigmas(args.steps)

                extra_args = {
                    "cond": cond,
                    "uncond": uncond,
                    "text_cfg_scale": args.cfg_text,
                    "image_cfg_scale": args.cfg_image,
                    "seg_cfg_scale": args.cfg_seg,
                }
                torch.manual_seed(seed)
                z = torch.randn_like(cond["c_concat1"][0]) * sigmas[0]
                z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
                x = model.decode_first_stage(z)
                x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
                x = 255.0 * rearrange(x, "1 c h w -> h w c")
                edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
            edited_image.save(os.path.join(args.output,file))
    


if __name__ == "__main__":
    main()
