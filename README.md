# Layered Diffusion Pipeline

**Note: The old README was moved to a [wiki page](https://github.com/nanashi161382/unstable_diffusion/wiki/Unstable-Diffusion-Pipeline:-a-wrapper-library-to-use-the-stable-diffusion-pipeline-more-easily).**

This is a wrapper library for the [stable diffusion pipeline](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/stable_diffusion) to allow us more flexibility in using [Stable Diffusion](https://stablediffusionweb.com/) and other derived models. The key concept of the pipeline is the **Layers** that stack up different prompts applied to a single image generation.

Unlike the old library, the new library was written from scratch, but the license follows the original stable diffusion pipeline for now. Please check the stable diffusion pipeline codebase for the license details.

## Basic Usage

The Layered Diffusion Pipeline can perform the same tasks as the original Stable Diffusion Pipeline does (except for the newer methods with specially tuned model, such as the new inpainting model and the depth to image model). This section explains how to use the library for the following three use cases.

* text to image
* image to image
* legacy inpainting

### Initialization

First, to use the library, you should put the python file in your current directory and then import like this.

```python
from pipeline_layered_diffusion import *
```

You can import the library in Google Colab like this.

```python
!pip install diffusers transformers scipy accelerate xformers safetensors omegaconf pytorch_lightning opencv-python
# Original code: https://github.com/nanashi161382/unstable_diffusion/blob/main/pipeline_layered_diffusion.py
!wget 'https://raw.githubusercontent.com/nanashi161382/unstable_diffusion/main/pipeline_layered_diffusion.py'
from pipeline_layered_diffusion import *
```

Then initialize the pipeline with a diffusers-format model data from a HuggingFace repository as follows.

```python
model_name = "stabilityai/stable-diffusion-2"
auth_token = "" # auth token for HuggingFace if needed
pipe = LayeredDiffusionPipeline().Connect(model_name, auth_token=auth_token)
```

If you have a diffusers-format model data locally, you can initialize the pipeline as follows.

```python
model_name = "stabilityai/stable-diffusion-2"  # This can be an arbitrary string.
model_path = "/path/to/model/directory"
pipe = LayeredDiffusionPipeline().Connect(model_name, cache_path=model_path)
```

If you have a StableDiffusion-style ckpt/safetensors file, you can also use it as follows.

```python
model_name = "stabilityai/stable-diffusion-2"  # This can be an arbitrary string.
model_path = "/path/to/model/directory/model_name.ckpt"
pipe = LayeredDiffusionPipeline().ConnectCkpt(model_name, checkpoint_path=model_path)
```

In this case, you can also swap the VAE as you like. Both the .pt-format data and the diffusers-format data are accepted.

```python
model_name = "stabilityai/stable-diffusion-2"  # This can be an arbitrary string.
model_path = "/path/to/model/directory/model_name.ckpt"
vae_path = "/path/to/vae/model/directory/vae_name.vae.pt"
pipe = LayeredDiffusionPipeline().ConnectCkpt(model_name, checkpoint_path=model_path, vae_path=vae_path)
```

### Image generation

Now you are ready for running the stable diffusion pipeline.

For text to image, you can go like this.

```python
image_size = (512, 512)  # width, height
image = pipe(
    num_steps=30,
    size=image_size,
    iterate=Layer(
        prompt="photo of orange pomeranian dog running in the park >>> cute face, fluffy",
        negative_prompt="bad quality, blur",
        cfg_scale=4.0
    ),
)[0]
display(image)
```
![text-to-image example](https://user-images.githubusercontent.com/118838049/209024330-8afee957-527c-4382-bbec-828e3bd1524a.png)

For image to image, here is what you need.

```python
image_size = (512, 512)  # width, height
image = pipe(
    num_steps=30,
    size=image_size,
    initialize=ByImage(
        image="init_image.png",
        strength=0.7,
    ),
    iterate=Layer(
        prompt="photo of orange pomeranian dog running in the park >>> cute face, fluffy",
        negative_prompt="bad quality, blur",
        cfg_scale=4.0
    ),
)[0]
display(image)
```

![image-to-image example](https://user-images.githubusercontent.com/118838049/209033208-bf3b7e1f-4f4d-4fad-ab4f-49b542e1d89e.png)


Finally, this is the code for legacy inpainting.

```python
image_size = (512, 512)  # width, height
image = pipe(
    num_steps=30,
    size=image_size,
    initialize=ByImage(
        image="init_image.png",
        strength=0.95,
    ),
    iterate=Layer(
        prompt="photo of orange tabby norwegian forest cat >>> cute face, fluffy",
        negative_prompt="bad quality, blur",
        cfg_scale=4.0
        mask_by="mask_image.png",
    ),
)[0]
display(image)
```

![inpainting example](https://user-images.githubusercontent.com/118838049/209058798-6898bfd7-a906-4079-89fd-ad8f5c5e3ffa.png)


### ShiftEncoding

By default, prompts and negative prompts are interpreted by ShiftEncoding that works differently from the stable diffusion pipeline.

ShiftEncoding is a newly proposed way of processing prompts in Stable Diffusion to enable the following functionalities.
* Eliminating position bias
* Dealing with long prompts
* Emphasizing words/phrases

For the details, please read "[ShiftEncoding to overcome position bias in Stable Diffusion prompts](https://github.com/nanashi161382/unstable_diffusion/wiki/ShiftEncoding-to-overcome-position-bias-in-Stable-Diffusion-prompts)."

To make it work as the original stable diffusion pipeline, you should just add `default_encoding=StandardEncoding()` to the pipeline as follows.

```python
image_size = (512, 512)  # width, height
image = pipe(
    num_steps=30,
    size=image_size,
    default_encoding=StandardEncoding(),
    initialize=Randomly(),
    iterate=Layer(
        prompt="black dog",
        negative_prompt="white cat",
        cfg_scale=7.5,
    ),
)[0]
display(image)
```

## Examples with layers

You can construct more complicated instruction for image generation with layers. Here are some code examples.

### Inpainting of 2+ objects with different prompts

The inpainting method repaints part of the original image with a single set of a prompt and a negative prompt. But there may be cases where you may want to repaint multiple parts of the original image at once. We can use multiple layers in the Layered Diffusion Pipeline to specify each object in each layer. Here is an example.

```python
image_size = (512, 512)  # width, height
image = pipe(
    num_steps=30,
    size=image_size,
    initialize=ByImage(
        image="init_image.png",
        strength=0.9,
    ),
    iterate=[
      Layer(
        prompt="orange puppy sitting on sofa",
        negative_prompt="bad quality",
        cfg_scale=4.0,
        mask_by="mask_image_left.png",
      ),
      Layer(
        prompt="red tulip flower in brown planter",
        negative_prompt="bad quality",
        cfg_scale=4.0,
        mask_by="mask_image_right.png",
      ),
    ]
)[0]
display(image)
```

![inpaint_with two_prompts example](https://user-images.githubusercontent.com/118838049/209110018-2bfd3abe-bc0c-44a7-9f06-7ff9f237294d.png)

This usually works, but sometimes different layers may affect each other especially when the prompts of the layers are similar. In such cases, we can specify layers as `distinct` to avoid interference. Here is an example.

```python
image_size = (512, 512)  # width, height
image = pipe(
    num_steps=30,
    size=image_size,
    initialize=ByImage(
        image="init_image.png",
        strength=0.9,
    ),
    iterate=[
      Layer(
        prompt="orange puppy sitting on sofa",
        negative_prompt="bad quality",
        cfg_scale=4.0,
        mask_by="mask_image_left.png",
        is_distinct=True,
      ),
      Layer(
        prompt="orange puppy sitting on sofa",
        negative_prompt="bad quality",
        cfg_scale=4.0,
        mask_by="mask_image_right.png",
        is_distinct=True,
      ),
    ]
)[0]
display(image)
```

![inpaint with 2 prompts by distinct layers example](https://user-images.githubusercontent.com/118838049/209110111-c9733f24-47cb-4249-a3b0-c6d7d17fc77f.png)

### Text to image with multiple different prompts

Layers also enables us to use different prompts for different part of the image, such as background and foreground. This is conceptually similar to inpainting, but instead of using a provided image as a background, it also uses text to image for generating the background image. Here is an example.

```python
image_size = (512, 512)  # width, height
image = pipe(
    num_steps=30,
    size=image_size,
    initialize=Randomly(),
    iterate=[
      Layer(
        prompt="photo of >>> green grass field, blue sky, mountain on horizon",
        negative_prompt="blur",
        cfg_scale=4.0,
      ),
      Layer(
        prompt="white persian cat >>> round face, blue eyes",
        negative_prompt="bad quality, malformed",
        cfg_scale=4.0,
        mask_by="mask_rect.png",
      ),
    ]
)[0]
display(image)
```

![txt2img 2 layers example](https://user-images.githubusercontent.com/118838049/209060625-5e1dfda5-6557-41a7-89e9-2d83cd5e399f.png)

Similar to inpainting above, it is also allowed to use 2+ layers for the foreground objects, and `is_distinct=True` is available as well. You should apply `is_distinct=True` only to the foreground layers. Here is an example with `is_distinct=True`. It makes each layer clearer but sometimes less integrated.

![txt2img 2 distinctt layers example](https://user-images.githubusercontent.com/118838049/209060714-54fb6cae-5587-4e67-bcb7-1a38008d0507.png)

### Image to image with multiple different prompts and strengths

In image to image method, `strength` defines how much we expect the original image to be changed in the final output. The strength value applies to the entire image. However we may want to apply different strengths to different parts of the image, such as background and foreground. Layers offer the ability for that.

When you want to apply lower strength to the background with a different prompt, you can set up the pipeline as follows.

```python
image_size = (512, 512)  # width, height
image = pipe(
    num_steps=30,
    size=image_size,
    initialize=ByImage(
        image="init_image.png",
        strength=0.8,
    ),
    iterate=[
        Layer(
            prompt="green grass, blue sky",
            negative_prompt="bad quality, brown dirt, white cloud, red surface",
            cfg_scale=8.0,
            strength=0.32
        ),
        Layer(
            prompt="white persian cat >>> round face, blue eyes, full body",
            negative_prompt="bad quality, malformed",
            cfg_scale=4.0,
            mask_by="mask_rect.png",
        ),
    ]
)[0]
display(image)
```

This example applies the strength 0.32 to the background (green field, blue sky) while applying the strength 0.8 to the foreground (white persian cat).

![two strengths example](https://user-images.githubusercontent.com/118838049/209082049-651265f2-b94c-4fd2-8ed8-7f8814b9d730.png)
