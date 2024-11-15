import functions_framework
import torch
import diffusers
from diffusers import StableDiffusionPipeline, AutoencoderKL
import random

device = "cuda"


def txt2img(prompt, negative_prompt, model_filename, vae_filename, height, width, steps, guidance, clip_skip, seed):

    pipe = StableDiffusionPipeline.from_single_file(model_filename, torch_dtype=torch.float16, safety_checker=None)

    if vae_filename is not None:
        pipe.vae = AutoencoderKL.from_single_file(vae_filename, torch_dtype=torch.float16)

    pipe = pipe.to(device)
    pipe.safety_checker = None

    # pipe.enable_attention_slicing()

    pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,
                                                      use_karras_sigmas=True)

    pipe.load_textual_inversion("/vol1/embeddings", weight_name="ng_deepnegative_v1_75t.pt", token="ng_deepnegative_v1_75t")
    pipe.load_textual_inversion("/vol1/embeddings", weight_name="bad-hands-5.pt", token="bad-hands-5")

    generator = torch.Generator(device=device).manual_seed(seed)

    images = pipe(prompt=prompt,
                  negative_prompt=negative_prompt,
                  height=height,
                  width=width,
                  num_inference_steps=steps,
                  guidance_scale=guidance,
                  clip_skip=clip_skip,
                  generator=generator,
                  ).images

    return images


@functions_framework.http
def main_handle(request):
    request_json = request.get_json(silent=True)

    prompt = request_json['prompt']
    negative_prompt = "ng_deepnegative_v1_75t, bad-hands-5, nsfw, sexy, breast, nude, 2 heads, duplicate, blurry, abstract, disfigured, deformed, framed, bad art, poorly drawn, extra limbs, b&w, weird colors, watermark, blur haze, long neck, elongated body, cropped image, out of frame, draft, deformed hands, twisted fingers, double image, malformed hands, multiple heads, ugly, poorly drawn hands, missing limb, cut-off, over satured, grain, lowres, bad anatomy, poorly drawn face, mutation, mutated, floating limbs, disconnected limbs, out of focus, long body, disgusting, extra fingers, missing arms, mutated hands, cloned face, missing legs,"

    model_filename = "/vol1/ckpts/beautifulRealistic_v60.safetensors"
    vae_filename = "/vol1/vae/vaeFtMse840000EmaPruned_vae.safetensors"

    height = 768
    width = 512
    steps = 20
    guidance = 7
    clip_skip = 2

    seed = random.randint(1, 4294967295)

    images = txt2img(prompt, negative_prompt, model_filename, vae_filename, height, width, steps, guidance, clip_skip, seed)
    images[0].save("/vol1/output/" + str(seed) + ".png")

    headers = {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
    }
    text = '{"result":"' + str(seed) + '.png"}'
    return (text, 200, headers)
