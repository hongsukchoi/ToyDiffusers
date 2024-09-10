# Source: https://huggingface.co/docs/diffusers/main/en/using-diffusers/weighted_prompts#weighting
# Source: https://github.com/damian0815/compel

from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch
from compel import Compel

import cv2

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_safetensors=True)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

prompt = '["a red cat", "playing with a", "ball"].and(0.6, 1, 2)'

generator = torch.Generator(device="cpu").manual_seed(33)

compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
prompt_embeds = compel_proc(prompt)


# image = pipe(prompt, generator=generator, num_inference_steps=20).images[0]
image = pipe(prompt_embeds=prompt_embeds, generator=generator, num_inference_steps=20).images[0]

image.save('cat4.png')


