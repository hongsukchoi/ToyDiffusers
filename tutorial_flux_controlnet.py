# Source: https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux
# Source: https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet_flux
# This Flux model doesn't really work with non-square images (09.06.2024) - Hongsuk Choi

import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline
from diffusers import FluxControlNetModel

import numpy as np
import cv2
from PIL import Image

base_model = "black-forest-labs/FLUX.1-dev" #"black-forest-labs/FLUX.1-schnell" #  
controlnet_model = "InstantX/FLUX.1-dev-controlnet-canny"
controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
pipe = FluxControlNetPipeline.from_pretrained(
    base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
)
# to run on low vram GPUs (i.e. between 4 and 32 GB VRAM)
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

pipe.to(torch.float16) # casting here instead of in the pipeline constructor because doing so in the constructor loads all models into CPU memory at once
# pipe.to("cuda")


# Get Control
# control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg")

# width, height = original_image.size
# Calculate new dimensions (round down to nearest divisible by 8)
# new_width = (width // 8) * 8
# new_height = (height // 8) * 8

# Crop the image (from top-left corner)
# original_image = original_image.crop((0, 0, new_width, new_height))
# width, height = original_image.size



# Cany
# original_image = load_image("")
# low_threshold = 100
# high_threshold = 200
# image = np.array(original_image)
# image = cv2.Canny(image, low_threshold, high_threshold)
# image = image[:, :, None]
# image = np.concatenate([image, image, image], axis=2)
# control_image = Image.fromarray(image)

# control_image = load_image("test_canny_edge.png")
control_image = load_image("finecontrolnet_detection.png")

prompt = "a blue sports car, a white helicopter, and a golden helicopter at a beach"
image = pipe(
    prompt,
    control_image=control_image,
    # height=height,
    # width=width,
    controlnet_conditioning_scale=0.6,
    num_inference_steps=35,
    guidance_scale=3,
).images[0]
image.save("test_carhelicopter_flux_control_canny_output.png")