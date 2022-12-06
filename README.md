# unstable_diffusion

**Note: Please use [release v3.0](https://github.com/nanashi161382/unstable_diffusion/tree/v3.0) for now because the latest codebase is under a big code change and is not compatible with this document.**

This is a library to use the [stable_diffusion pipeline](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/stable_diffusion) more easily for the interactive use cases. The official pipeline classes are OK for the batch use cases, but a bit cumbersome to use for the interactive use cases especially on Google Colab. Especially if you want to switch different pipeline types such as text-to-image, image-to-image and inpaint. This library works as a wrapper to the library to give better experiences for the purpose.

This library also aims to give additional functionalities for advanced users.

## [pipeline_unstable_diffusion.py](pipeline_unstable_diffusion.py)

An alternative stable_diffusion pipeline. Part of the code in the file is a copy of the stable_diffusion pipelines. Please check the license of the original code.

To use the library, you should put it in your current directory and then import like this.

```python
from pipeline_unstable_diffusion import (
    Txt2Img, Img2Img, Inpaint,
    ByLatents, Randomly, StandardEncoding, ShiftEncoding,
    UnstableDiffusionPipeline, ImageModel,
)
```

You can import the library in Google Colab like this.

```python
!pip install --upgrade diffusers transformers scipy accelerate
use_xformers = False
# Original code: https://github.com/nanashi161382/unstable_diffusion/blob/main/pipeline_unstable_diffusion.py
!wget 'https://raw.githubusercontent.com/nanashi161382/unstable_diffusion/main/pipeline_unstable_diffusion.py'
from pipeline_unstable_diffusion import (
    Txt2Img, Img2Img, Inpaint,
    ByLatents, Randomly, StandardEncoding, ShiftEncoding,
    UnstableDiffusionPipeline, ImageModel,
)
```

If you want to enable xformers memory efficient attention (probably only available for stable diffusion 2 series?), you can run this instead.
```python
!pip install --upgrade diffusers transformers scipy accelerate
# use the pre-release versions for triton
!pip install --upgrade --pre triton
!pip install -q https://github.com/metrolobo/xformers_wheels/releases/download/1d31a3ac_various_6/xformers-0.0.14.dev0-cp37-cp37m-linux_x86_64.whl
use_xformers = True
# Original code: https://github.com/nanashi161382/unstable_diffusion/blob/main/pipeline_unstable_diffusion.py
!wget 'https://raw.githubusercontent.com/nanashi161382/unstable_diffusion/main/pipeline_unstable_diffusion.py'
from pipeline_unstable_diffusion import (
    Txt2Img, Img2Img, Inpaint,
    ByLatents, Randomly, StandardEncoding, ShiftEncoding,
    UnstableDiffusionPipeline, ImageModel,
)
```

Then initialize the pipeline as follows.

```python
dataset = "Linaqruf/anything-v3.0"
auth_token = "" # auth token for HuggingFace if needed
pipe = UnstableDiffusionPipeline().Connect(dataset, auth_token=auth_token, use_xformers=use_xformers)
```

Now you are ready for running the stable diffusion pipeline.

For text to image, you can go like this.

```python
prompt = "1girl, 1boy"
negative_prompt = "1girl"
guidance_scale = 7.5
num_steps = 50
image_size = (512, 512)  # width, height
symmetric = False

image = pipe(
    pipeline_type=Txt2Img(
        initialize=Randomly(symmetric=symmetric),
        size=image_size,
    ),
    text_input=StandardEncoding(
        prompt=prompt,
        negative_prompt=negative_prompt,
    ),
    guidance_scale=guidance_scale,
    num_inference_steps=num_steps,
  )[0]
display(image)
```

For image to image, here is what you need.

```python
prompt = "1girl, 1boy"
negative_prompt = "1girl"
guidance_scale = 7.5
init_image = "init_image.png"
strength = 0.8
num_steps = 50
image_size = (512, 512)  # width, height

def OpenImage(filename):
    image = Image.open(filename).convert("RGB")
    image = image.resize(image_size)
    return image

image = pipe(
    pipeline_type=Img2Img(
        init_image=OpenImage(init_image),
        strength=strength,
    ),
    text_input=StandardEncoding(
        prompt=prompt,
        negative_prompt=negative_prompt,
    ),
    guidance_scale=guidance_scale,
    num_inference_steps=num_steps,
  )[0]
display(image)
```

Finally, this is the code for inpaint. (note: the inpaint logic is equivalent to `StableDiffusionInpaintPipelineLegacy`, but not `StableDiffusionInpaintPipeline`.)

```python
prompt = "1girl, 1boy"
negative_prompt = "1girl"
guidance_scale = 7.5
init_image = "init_image.png"
mask_image = "mask_image.png"
strength = 0.8
num_steps = 50
image_size = (512, 512)  # width, height

def OpenImage(filename):
    image = Image.open(filename).convert("RGB")
    image = image.resize(image_size)
    return image

image = pipe(
    pipeline_type=Inpaint(
        init_image=OpenImage(init_image),
        mask_image=OpenImage(mask_image),
        strength=strength,
    ),
    text_input=StandardEncoding(
        prompt=prompt,
        negative_prompt=negative_prompt,
    ),
    guidance_scale=guidance_scale,
    num_inference_steps=num_steps,
  )[0]
display(image)
```

They work almost the same as the original stable diffusion pipelines, but there are some differences.
Please check the description in the [initial commit](https://github.com/nanashi161382/unstable_diffusion/commit/7c94b3c74e7a23375e4158b54b85bbc6630302bf).

## ShiftEncoding

ShiftEncoding is a newly proposed way of processing prompts in Stable Diffusion to enable the following functionalities.
* Eliminating position bias
* Dealing with long prompts
* Emphasizing words/phrases

For the details, please read "[ShiftEncoding to overcome position bias in Stable Diffusion prompts](https://github.com/nanashi161382/unstable_diffusion/wiki/ShiftEncoding-to-overcome-position-bias-in-Stable-Diffusion-prompts)."

To use this, you should simply replace `StandardEncoding` with `ShiftEncoding` in the examples above. Here is an example for text to image.

```python
prompt = "1girl, 1boy"
negative_prompt = "1girl"
guidance_scale = 7.5
num_steps = 50
image_size = (512, 512)  # width, height
symmetric = False

image = pipe(
    pipeline_type=Txt2Img(
        initialize=Randomly(symmetric=symmetric),
        size=image_size,
    ),
    text_input=ShiftEncoding(
        prompt=prompt,
        negative_prompt=negative_prompt,
    ),
    guidance_scale=guidance_scale,
    num_inference_steps=num_steps,
  )[0]
display(image)
```
