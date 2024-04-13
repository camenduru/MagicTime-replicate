import os, sys
from cog import BasePredictor, Input, Path
sys.path.append('/content/MagicTime')
os.chdir('/content/MagicTime')

import copy
import time
import torch
import random
from glob import glob
from omegaconf import OmegaConf
from safetensors import safe_open
from diffusers import AutoencoderKL
from diffusers import DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

from utils.unet import UNet3DConditionModel
from utils.pipeline_magictime import MagicTimePipeline
from utils.util import save_videos_grid, convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint, load_diffusers_lora_unet, convert_ldm_clip_text_model

pretrained_model_path   = "./ckpts/Base_Model/stable-diffusion-v1-5"
inference_config_path   = "./sample_configs/RealisticVision.yaml"
magic_adapter_s_path    = "./ckpts/Magic_Weights/magic_adapter_s/magic_adapter_s.ckpt"
magic_adapter_t_path    = "./ckpts/Magic_Weights/magic_adapter_t"
magic_text_encoder_path = "./ckpts/Magic_Weights/magic_text_encoder"

device = torch.device('cuda:0')

class MagicTimeController:
    def __init__(self):
        # config dirs
        self.basedir                = os.getcwd()
        self.stable_diffusion_dir   = os.path.join(self.basedir, "ckpts", "Base_Model")
        self.motion_module_dir      = os.path.join(self.basedir, "ckpts", "Base_Model", "motion_module")
        self.personalized_model_dir = os.path.join(self.basedir, "ckpts", "DreamBooth")
        self.savedir                = os.path.join(self.basedir, "outputs")
        os.makedirs(self.savedir, exist_ok=True)

        self.dreambooth_list    = []
        self.motion_module_list = []
        
        self.selected_dreambooth    = None
        self.selected_motion_module = None
        
        self.refresh_motion_module()
        self.refresh_personalized_model()
        
        # config models
        self.inference_config      = OmegaConf.load(inference_config_path)[1]

        self.tokenizer             = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        self.text_encoder          = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").to(device)
        self.vae                   = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(device)
        self.unet                  = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(self.inference_config.unet_additional_kwargs)).to(device)
        self.text_model            = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        self.unet_model            = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(self.inference_config.unet_additional_kwargs))

        self.update_motion_module(self.motion_module_list[0])
        self.update_motion_module_2(self.motion_module_list[0])
        self.update_dreambooth(self.dreambooth_list[0])
        
    def refresh_motion_module(self):
        motion_module_list = glob(os.path.join(self.motion_module_dir, "*.ckpt"))
        self.motion_module_list = [os.path.basename(p) for p in motion_module_list]

    def refresh_personalized_model(self):
        dreambooth_list = glob(os.path.join(self.personalized_model_dir, "*.safetensors"))
        self.dreambooth_list = [os.path.basename(p) for p in dreambooth_list]

    def update_dreambooth(self, dreambooth_dropdown, motion_module_dropdown=None):
        self.selected_dreambooth = dreambooth_dropdown
        
        dreambooth_dropdown = os.path.join(self.personalized_model_dir, dreambooth_dropdown)
        dreambooth_state_dict = {}
        with safe_open(dreambooth_dropdown, framework="pt", device="cpu") as f:
            for key in f.keys(): dreambooth_state_dict[key] = f.get_tensor(key)
                
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(dreambooth_state_dict, self.vae.config)
        self.vae.load_state_dict(converted_vae_checkpoint)

        del self.unet
        self.unet = None
        torch.cuda.empty_cache()
        time.sleep(1)
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(dreambooth_state_dict, self.unet_model.config)
        self.unet = copy.deepcopy(self.unet_model)
        self.unet.load_state_dict(converted_unet_checkpoint, strict=False)

        del self.text_encoder
        self.text_encoder = None
        torch.cuda.empty_cache()
        time.sleep(1)
        text_model = copy.deepcopy(self.text_model)
        self.text_encoder = convert_ldm_clip_text_model(text_model, dreambooth_state_dict)

        from swift import Swift
        magic_adapter_s_state_dict = torch.load(magic_adapter_s_path, map_location="cpu")
        self.unet = load_diffusers_lora_unet(self.unet, magic_adapter_s_state_dict, alpha=1.0)
        self.unet = Swift.from_pretrained(self.unet, magic_adapter_t_path)
        self.text_encoder = Swift.from_pretrained(self.text_encoder, magic_text_encoder_path)

        return None

    def update_motion_module(self, motion_module_dropdown):
        self.selected_motion_module = motion_module_dropdown
        motion_module_dropdown = os.path.join(self.motion_module_dir, motion_module_dropdown)
        motion_module_state_dict = torch.load(motion_module_dropdown, map_location="cpu")
        _, unexpected = self.unet.load_state_dict(motion_module_state_dict, strict=False)
        assert len(unexpected) == 0
        return None
    
    def update_motion_module_2(self, motion_module_dropdown):
        self.selected_motion_module = motion_module_dropdown
        motion_module_dropdown = os.path.join(self.motion_module_dir, motion_module_dropdown)
        motion_module_state_dict = torch.load(motion_module_dropdown, map_location="cpu")
        _, unexpected = self.unet_model.load_state_dict(motion_module_state_dict, strict=False)
        assert len(unexpected) == 0
        return None

    def magictime(
        self,
        dreambooth_dropdown,
        motion_module_dropdown,
        prompt_textbox,
        negative_prompt_textbox,
        width_slider,
        height_slider,
        seed_textbox,
    ):
        torch.cuda.empty_cache()
        time.sleep(1)

        if self.selected_motion_module != motion_module_dropdown: self.update_motion_module(motion_module_dropdown)
        if self.selected_motion_module != motion_module_dropdown: self.update_motion_module_2(motion_module_dropdown)
        if self.selected_dreambooth != dreambooth_dropdown: self.update_dreambooth(dreambooth_dropdown)
        
        while self.text_encoder is None or self.unet is None:
            self.update_dreambooth(dreambooth_dropdown, motion_module_dropdown)

        if is_xformers_available(): self.unet.enable_xformers_memory_efficient_attention()

        pipeline = MagicTimePipeline(
            vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer, unet=self.unet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs))
        ).to(device)     

        if int(seed_textbox) > 0: seed = int(seed_textbox)
        else: seed = random.randint(1, 1e16)
        torch.manual_seed(int(seed))
        
        assert seed == torch.initial_seed()
        print(f"### seed: {seed}")
        
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        
        sample = pipeline(
            prompt_textbox,
            negative_prompt     = negative_prompt_textbox,
            num_inference_steps = 25,
            guidance_scale      = 8.,
            width               = width_slider,
            height              = height_slider,
            video_length        = 16,
            generator           = generator,
        ).videos

        save_sample_path = os.path.join(self.savedir, f"sample.mp4")
        save_videos_grid(sample, save_sample_path)
    
        json_config = {
            "prompt": prompt_textbox,
            "n_prompt": negative_prompt_textbox,
            "width": width_slider,
            "height": height_slider,
            "seed": seed,
            "dreambooth": dreambooth_dropdown,
        }

        torch.cuda.empty_cache()
        time.sleep(1)
        return save_sample_path, json_config

class Predictor(BasePredictor):
    def setup(self) -> None:
        self.controller = MagicTimeController()
    def predict(
        self,
        dreambooth: str = Input(choices=["ToonYou_beta6.safetensors", "RealisticVisionV60B1_v51VAE.safetensors", "RcnzCartoon.safetensors"], default="ToonYou_beta6.safetensors"),
        # motion_module: str = Input(default="motion_module.ckpt"),
        prompt: str = Input(default="Bean sprouts grow and mature from seeds."),
        negative_prompt: str = Input(default="worst quality, low quality, letterboxed"),
        width: int = Input(default=512),
        height: int = Input(default=512),
        seed: str = Input(default="1496541313"),
    ) -> Path:
        output_video, json_config = self.controller.magictime(dreambooth_dropdown=dreambooth, 
                                    motion_module_dropdown="motion_module.ckpt", 
                                    prompt_textbox=prompt, 
                                    negative_prompt_textbox=negative_prompt, 
                                    width_slider=width, height_slider=height, seed_textbox=seed)
        return Path(output_video)