# Source: https://huggingface.co/docs/diffusers/main/en/using-diffusers/weighted_prompts#weighting
# Source: https://huggingface.co/docs/diffusers/main/en/using-diffusers/controlnet
# Source: https://github.com/damian0815/compel
import argparse

import torch
import cv2
import numpy as np
from PIL import Image

from controlnet_aux import OpenposeDetector
from diffusers import DiffusionPipeline, StableDiffusionXLControlNetPipeline, UniPCMultistepScheduler, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
from compel import Compel, ReturnedEmbeddingsType

if __name__ == '__main__':
    # For 2
    prompt = '["a (firefighter)2.0 on the left", "a (doctor)2.0 on the right", "in the rainy forest"].and()'

    # For 3
    # prompt = '["From left to right: ", "a doctor", "a firefighter", "and a medieval queen", "in outer space"].and(1.0, 1.5, 1.5, 1.5, 1.0)'

    # For 7 
    # prompt = '["From left to right: ", "a medieval queen", "a doctor", "a firefighter", "a man with white shirts"\
        # "a woman with green pants", "a clown", "a laboratory engineer", "in a bunker"].and(1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.0)'


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
    openpose_image.save(f'{test_input[:-4]}_sdxl_control_openpose_input.png')
    
    controlnet = ControlNetModel.from_pretrained(
            "thibaud/controlnet-openpose-sdxl-1.0", 
            torch_dtype=torch.float16,
            use_safetensors=False
    )

    # vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
    # pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0",
    #     controlnet=controlnet,
    #     vae=vae,
    #     torch_dtype=torch.float16,
    #     use_safetensors=True
    # )

    # pipe.enable_model_cpu_offload()

    # I (Hongsuk Choi) think ControlNet doesn't apply
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        variant="fp16", 
        # controlnet=controlnet,
        use_safetensors=True, 
        torch_dtype=torch.float16).to("cuda")
    
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to('cuda')
    generator = torch.Generator(device='cuda').manual_seed(1)
    compel = Compel(device='cuda', tokenizer=[pipe.tokenizer, pipe.tokenizer_2] , text_encoder=[pipe.text_encoder, pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
    conditioning, pooled = compel(prompt)

    image = pipe(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, image=openpose_image, generator=generator, num_inference_steps=30).images[0]
    # image = pipe(prompt, image=openpose_image, generator=generator, num_inference_steps=30).images[0]

    image.save(f'{test_input[:-4]}_sdxl_posecontrol_promptweight_output.png')
