# Source: https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux
# Source: https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union
import argparse
import torch
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image, make_image_grid
from diffusers import FluxControlNetPipeline, FluxControlNetModel, FluxMultiControlNetModel

import random
import cv2
import numpy as np
from PIL import Image


if __name__ == '__main__':
    
    count = 30
    for i in range(count):
        seed = random.randint(1,100000000)
    
        prompt = "From left to right: a metal humanoid, a darth vader, and a medieval knight in the desert"

        # Create an ArgumentParser object
        parser = argparse.ArgumentParser(description="Process test_input argument.")
        
        # Add argument 'test_input'
        parser.add_argument('--test_input', type=str, help="Input for the test.")
        
        # Parse the arguments
        args = parser.parse_args()

        test_input = args.test_input
        # OpenPose Detector
        openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        original_image = load_image(test_input)

        openpose_image = openpose(original_image)
        openpose_image.save(f'{test_input[:-4]}_flux_control_openpose_input.png')

        control_image_pose = openpose_image
        control_mode_pose = 4

        base_model = 'black-forest-labs/FLUX.1-dev'
        controlnet_model_union = 'InstantX/FLUX.1-dev-Controlnet-Union'

        controlnet_union = FluxControlNetModel.from_pretrained(controlnet_model_union, torch_dtype=torch.bfloat16)
        controlnet = FluxMultiControlNetModel([controlnet_union]) # we always recommend loading via FluxMultiControlNetModel

        pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
        # pipe.to("cuda")
        # to run on low vram GPUs (i.e. between 4 and 32 GB VRAM)
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        pipe.to(torch.float16) # casting here instead of in the pipeline constructor because doing so in the constructor loads all models into CPU memory at once
        image = pipe(
            prompt, 
            control_image=[control_image_pose],
            control_mode=[control_mode_pose],
            # width=width,
            # height=height,
            controlnet_conditioning_scale=[0.5],
            num_inference_steps=20, 
            guidance_scale=3.5,
            generator=torch.manual_seed(seed),
        ).images[0]

        image.save(f'{test_input[:-4]}_flux_posecontrol_output_{i}.png')

