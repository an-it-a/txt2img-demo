import functions_framework
import torch
import diffusers
from diffusers import StableDiffusionPipeline, AutoencoderKL
import random

device = "cuda"

scheduler = {
    "DPM++ 2M": lambda pipe: diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config),
    "DPM++ 2M Karras": lambda pipe: diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,
                                                                                      use_karras_sigmas=True),
    "DPM++ 2M SDE": lambda pipe: diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,
                                                                                   algorithm_type="sde-dpmsolver++"),
    "DPM++ 2M SDE Karras": lambda pipe: diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,
                                                                                          algorithm_type="sde-dpmsolver++",
                                                                                          use_karras_sigmas=True),
    "DPM++ SDE": lambda pipe: diffusers.DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config),
    "DPM++ SDE Karras": lambda pipe: diffusers.DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config,
                                                                                        use_karras_sigmas=True),
    "DPM2": lambda pipe: diffusers.KDPM2DiscreteScheduler.from_config(pipe.scheduler.config),
    "DPM2 Karras": lambda pipe: diffusers.DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config,
                                                                                   use_karras_sigmas=True),
    "DPM2 a": lambda pipe: diffusers.KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config),
    "DPM2 a Karras": lambda pipe: diffusers.KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config,
                                                                                        use_karras_sigmas=True),
    "Euler": lambda pipe: diffusers.EulerDiscreteScheduler.from_config(pipe.scheduler.config),
    "Euler a": lambda pipe: diffusers.EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config),
    "Heun": lambda pipe: diffusers.HeunDiscreteScheduler.from_config(pipe.scheduler.config),
    "LMS": lambda pipe: diffusers.LMSDiscreteScheduler.from_config(pipe.scheduler.config),
    "LMS Karras": lambda pipe: diffusers.LMSDiscreteScheduler.from_config(pipe.scheduler.config,
                                                                          use_karras_sigmas=True),
    "UniPCMultistep": lambda pipe: diffusers.UniPCMultistepScheduler.from_config(pipe.scheduler.config),
    "LCM": lambda pipe: diffusers.LCMScheduler.from_config(pipe.scheduler.config),
}


def get_pipieline(pipeline_type, model_filename, vae_filename, scheduler_name):
    pipe = pipeline_type.from_single_file(model_filename, torch_dtype=torch.float16, safety_checker=None)

    if vae_filename is not None:
        pipe.vae = AutoencoderKL.from_single_file(vae_filename, torch_dtype=torch.float16)

    pipe = pipe.to(device)
    pipe.safety_checker = None

    # pipe.enable_attention_slicing()

    pipe.scheduler = scheduler[scheduler_name](pipe)

    return pipe


def txt2img(prompt, negative_prompt, model_filename, vae_filename, scheduler_name, height, width, steps, guidance, clip_skip, seed):

    pipe = get_pipieline(StableDiffusionPipeline, model_filename, vae_filename, scheduler_name)

    pipe.load_textual_inversion("/vol1/embeddings", weight_name="easynegative.safetensors", token="easynegative")
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
    negative_prompt = "worst quality, low quality, bad face, ugly face, nsfw"

    model_filename = "/vol1/ckpts/beautifulRealistic_v60.safetensors"
    vae_filename = "/vol1/vae/vaeFtMse840000EmaPruned_vae.safetensors"

    scheduler_name = "DPM++ 2M Karras"

    height = 768
    width = 512
    steps = 20
    guidance = 7
    clip_skip = 2

    seed = random.randint(1, 4294967295)

    images = txt2img(prompt, negative_prompt, model_filename, vae_filename, scheduler_name, height, width, steps, guidance, clip_skip, seed)
    images[0].save("/vol1/output/" + str(seed) + ".png")

    return 'Result: {}.png'.format(seed)
