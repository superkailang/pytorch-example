# ! pip install diffusers==0.14.0 xformers transformers scipy ftfy accelerate controlnet_aux

import numpy as np
import torch
from PIL import Image
from controlnet_aux import HEDdetector
from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image



def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


def generate_condition(image):
    hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
    hed_image = hed(image)
    print(hed_image)


def generate_mask(org_image):
    mask_img = np.array(org_image.convert("RGB"))
    mask_img[mask_img == 0] = 255
    mask_img[mask_img < 255] = 0
    return Image.fromarray(mask_img)


def merge_image(a_image, b_image):
    img3 = Image.blend(a_image,b_image, 0.3)
    return img3


shape = (512, 512)
light_image = load_image(
    "image/light.jpg"
)
init_image = load_image(
    "image/export.png"
)
init_image = init_image.resize(shape)
light_image = light_image.resize(shape)
mask_image = generate_mask(init_image)
mask_image = mask_image.resize(shape)

merge_image(init_image, light_image)

edge_image = load_image("image/edge_mask.png")
# generate_condition(edge_image)

# generator = torch.Generator(device="cpu").manual_seed(1)
# mask_image = load_image(
#     "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy_mask.png"
# )


# pipe_sd = StableDiffusionInpaintPipeline.from_pretrained(
#     "runwayml/stable-diffusion-inpainting",
#     revision="fp16",
#     torch_dtype=torch.float16,
# )
# # speed up diffusion process with faster scheduler and memory optimization
# pipe_sd.scheduler = UniPCMultistepScheduler.from_config(pipe_sd.scheduler.config)
# # remove following line if xformers is not installed
# pipe_sd.enable_xformers_memory_efficient_attention()
# pipe_sd.to('cuda')

# control_image = make_inpaint_condition(init_image, mask_image)

controlnet = ControlNetModel.from_pretrained(
    "fusing/stable-diffusion-v1-5-controlnet-hed", torch_dtype=torch.float16
)

controlnet = ControlNetModel.from_pretrained(
    "fusing/stable-diffusion-v1-5-controlnet-hed", torch_dtype=torch.float16
)



pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16
)

pipe.controlnet = [controlnet, depth_controlnet]

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
pipe.enable_xformers_memory_efficient_attention()

pipe.to('cuda')

# generate image
generator = torch.manual_seed(0)
new_image = pipe(
    prompt="a wine bottle on the top blue platform",
    num_inference_steps=20,
    generator=generator,
    image=init_image,
    control_image=edge_image,
    mask_image=mask_image
).images[0]

# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# pipe.enable_model_cpu_offload()

# generate image
# image = pipe(
#     "a handsome man with ray-ban sunglasses",
#     num_inference_steps=20,
#     generator=generator,
#     eta=1.0,
#     image=init_image,
#     mask_image=mask_image,
#     control_image=control_image,
# ).images[0]
