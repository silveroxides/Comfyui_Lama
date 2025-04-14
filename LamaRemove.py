import os
import sys
import torch
from omegaconf import OmegaConf
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))

MODELS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config")


class LamaApply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "lama":("LAMA",),
                "config":("YAML_CONFIG",),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "lama_remove"

    CATEGORY = "lama"
    
    def lama_remove(self,image,mask,config,lama):     
        device = "cuda" if torch.cuda.is_available() else "cpu"

        img_inpainted = self.inpaint_img_with_lama(
            image, mask, config, lama, device=device)
        img = torch.from_numpy(img_inpainted)[None,]
        return (img,)
    
    def inpaint_img_with_lama(
        self,
        img,
        mask,
        config_p: OmegaConf,
        model,
        mod=8,
        device="cuda"
    ):
        from saicinpainting.evaluation.utils import move_to_device
        from saicinpainting.evaluation.data import pad_tensor_to_modulo
        batch = {}
        print(img.shape)
        print(mask.shape)
        mask=mask*255
        batch['image'] = img.permute(0,3, 1, 2)
        batch['mask'] = mask[None, None]
        unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
        batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
        batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
        batch = move_to_device(batch, device)
        batch['mask'] = (batch['mask'] > 0) * 1

        batch = model(batch)
        cur_res = batch[config_p.out_key][0].permute(1, 2, 0)
        cur_res = cur_res.detach().cpu().numpy()

        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

        #cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        return cur_res
    
    
class LamaModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "config":("YAML_CONFIG",),
            },
        }

    RETURN_TYPES = ("LAMA","YAML_CONFIG")

    FUNCTION = "load_lama"

    CATEGORY = "lama"
    def load_lama(self,config):
        from saicinpainting.training.trainers import load_checkpoint
        device = torch.device(config.device)
        config.training_model.predict_only = True
        config.visualizer.kind = 'noop'
        checkpoint_path = os.path.join(MODELS_DIR,config.model.checkpoint)
        if not os.path.exists(checkpoint_path) and os.path.exists("/stable-diffusion-cache/models/inpaint"):
            checkpoint_path = "/stable-diffusion-cache/models/inpaint/big-lama.ckpt"
        model = load_checkpoint(config, checkpoint_path, strict=False)
        model.to(device)
        model.freeze()
        return (model,config)
        

class YamlConfigLoader:

    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(CONFIG_DIR) if f.endswith('.yaml')]
        return {
            "required":{
                "yaml_config": (files,),
            },
        }

    RETURN_TYPES = ("YAML_CONFIG",)

    FUNCTION = "load_yaml"

    CATEGORY = "load_yaml"

    def load_yaml(self,yaml_config):
        yaml_path=os.path.join(CONFIG_DIR, yaml_config)
        config = OmegaConf.load(yaml_path)
        return (config,)



# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LamaModelLoader":LamaModelLoader,
    "LamaApply": LamaApply,
    "YamlConfigLoader":YamlConfigLoader,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LamaModelLoader":"LamaModelLoader",
    "LamaApply": "LamaApply",
    "YamlConfigLoader":"YamlConfigLoader"
}
