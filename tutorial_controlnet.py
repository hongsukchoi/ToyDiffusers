# Source: https://huggingface.co/docs/diffusers/main/en/using-diffusers/controlnet
# I (Hongsuk Choi) think this tutorial (https://huggingface.co/lllyasviel/sd-controlnet-openpose) is no longer valid, because RUNWAY removed the repo from diffusers.

from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
import torch

import cv2
import numpy as np
from PIL import Image

# OpenPose Detector
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
original_image = load_image('twoperson_input.jpg')
# "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png"

openpose_image = openpose(original_image)

# Cany
# low_threshold = 100
# high_threshold = 200
# image = np.array(original_image)
# image = cv2.Canny(image, low_threshold, high_threshold)
# image = image[:, :, None]
# image = np.concatenate([image, image, image], axis=2)
# canny_image = Image.fromarray(image)

# Controlnet and SD
# controlnet = ControlNetModel.from_pretrained(
#     "diffusers/controlnet-canny-sdxl-1.0",
#     torch_dtype=torch.float16,
#     use_safetensors=True
# )
controlnet = ControlNetModel.from_pretrained(
        "thibaud/controlnet-openpose-sdxl-1.0", 
        torch_dtype=torch.float16,
        use_safetensors=False
)

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe.enable_model_cpu_offload()



image = pipe("a doctor and a firefighter", image=openpose_image, num_inference_steps=20).images[0]

image.save('doctor_and_firefighter.png')