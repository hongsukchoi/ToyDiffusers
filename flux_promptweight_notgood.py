# Source: https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux
# Source: https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet_flux

# I (Hongsuk Choi) think there's no solution yet (09.04.2024)
# https://github.com/damian0815/compel/issues/99
# Maybe https://github.com/xhinker/sd_embed , but not flexible to set weights for each word
import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline
from diffusers import FluxControlNetModel
from compel import Compel, ReturnedEmbeddingsType

import numpy as np
import cv2
from PIL import Image

# Get Control
original_image = load_image("twoperson_input.jpg")

# Cany
low_threshold = 100
high_threshold = 200
image = np.array(original_image)
image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
control_image = Image.fromarray(image)

base_model = "black-forest-labs/FLUX.1-dev" #"black-forest-labs/FLUX.1-schnell" #  
controlnet_model = "InstantX/FLUX.1-dev-controlnet-canny"
controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
pipe = FluxControlNetPipeline.from_pretrained(
    base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
)


# pipe.to("cuda")
# to run on low vram GPUs (i.e. between 4 and 32 GB VRAM)
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
pipe.to(torch.float16) # casting here instead of in the pipeline constructor because doing so in the constructor loads all models into CPU memory at once



prompt = "a doctor++ and a firefighter++"
# compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
# prompt_embeds = compel_proc(prompt)
compel = Compel(tokenizer=[pipe.tokenizer, pipe.tokenizer_2] , text_encoder=[pipe.text_encoder, pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
conditioning, pooled = compel(prompt)
# compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=True)
# conditioning, pooled = compel(prompt)


image = pipe(
    prompt_embeds=conditioning, pooled_prompt_embeds=pooled,
    # prompt_embeds=prompt_embeds,
    control_image=control_image,
    controlnet_conditioning_scale=0.6,
    num_inference_steps=28,
    guidance_scale=3.5,
).images[0]
image.save("flux.png")