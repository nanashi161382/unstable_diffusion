# Layered Diffusion Pipeline

**Note: The old README was moved to a [wiki page](https://github.com/nanashi161382/unstable_diffusion/wiki/Unstable-Diffusion-Pipeline:-a-wrapper-library-to-use-the-stable-diffusion-pipeline-more-easily).**

This is a wrapper library for the [stable diffusion pipeline](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/stable_diffusion) to allow us more flexibility in using [Stable Diffusion](https://stablediffusionweb.com/) and other derived models. The key concept of the pipeline is the **Layers** that stack up different prompts applied to a single image generation.

Unlike the old library, the new library was written from scratch, but the license follows the original stable diffusion pipeline for now. Please check the stable diffusion pipeline codebase for the license details.

## Basic Usage

The Layered Diffusion Pipeline can perform the same tasks as the original Stable Diffusion Pipeline does (except for the newer methods with specially tuned model, such as the new inpainting model and the depth to image model). This section explains how to use the library for the following three use cases.

* text to image
* image to image
* legacy inpainting

First, to use the library, you should put the python file in your current directory and then import like this.

```python
from pipeline_layered_diffusion import (
    StandardEncoding, ShiftEncoding,
    Randomly, ByLatents, ByImage, ByBothOf,
    ComboOf, ScaledMask, UnionMask, IntersectMask,
    SetDebugLevel, OpenImage,
    Layer, LayeredDiffusionPipeline,
    ImageModel, TextModel, SharedTarget,
)
```

You can import the library in Google Colab like this.

```python
!pip install --upgrade diffusers transformers scipy accelerate
use_xformers = False
# Original code: https://github.com/nanashi161382/unstable_diffusion/blob/main/pipeline_layered_diffusion.py
!wget 'https://raw.githubusercontent.com/nanashi161382/unstable_diffusion/main/pipeline_layered_diffusion.py'
from pipeline_layered_diffusion import (
    StandardEncoding, ShiftEncoding,
    Randomly, ByLatents, ByImage, ByBothOf,
    ComboOf, ScaledMask, UnionMask, IntersectMask,
    SetDebugLevel, OpenImage,
    Layer, LayeredDiffusionPipeline,
    ImageModel, TextModel, SharedTarget,
)
```

If you want to enable xformers memory efficient attention, you can run this instead.

```python
!pip install --upgrade diffusers transformers scipy accelerate
# use the pre-release versions for triton
!pip install --upgrade --pre triton
!pip install -q https://github.com/metrolobo/xformers_wheels/releases/download/1d31a3ac_various_6/xformers-0.0.14.dev0-cp37-cp37m-linux_x86_64.whl
use_xformers = True
# Original code: https://github.com/nanashi161382/unstable_diffusion/blob/main/pipeline_layered_diffusion.py
!wget 'https://raw.githubusercontent.com/nanashi161382/unstable_diffusion/main/pipeline_layered_diffusion.py'
from pipeline_layered_diffusion import (
    StandardEncoding, ShiftEncoding,
    Randomly, ByLatents, ByImage, ByBothOf,
    ComboOf, ScaledMask, UnionMask, IntersectMask,
    SetDebugLevel, OpenImage,
    Layer, LayeredDiffusionPipeline,
    ImageModel, TextModel, SharedTarget,
)
```

Then initialize the pipeline as follows.

```python
dataset = "stabilityai/stable-diffusion-2"
auth_token = "" # auth token for HuggingFace if needed
pipe = UnstableDiffusionPipeline().Connect(dataset, auth_token=auth_token, use_xformers=use_xformers)
```

Now you are ready for running the stable diffusion pipeline.

For text to image, you can go like this.

```python
image_size = (512, 512)  # width, height
image = pipe(
    num_steps=30,
    initialize=Randomly(),
    size=image_size,
    layers=Layer(
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
    initialize=ByImage(
        image="init_image.png",
        strength=0.7,
    ),
    size=image_size,
    layers=Layer(
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
    initialize=ByImage(
        image="init_image.png",
        strength=0.95,
    ),
    size=image_size,
    layers=Layer(
        prompt="photo of orange tabby norwegian forest cat >>> cute face, fluffy",
        negative_prompt="bad quality, blur",
        cfg_scale=4.0
        mask_by="mask_image.png",
    ),
)[0]
display(image)
```

![inpainting example](https://user-images.githubusercontent.com/118838049/209058798-6898bfd7-a906-4079-89fd-ad8f5c5e3ffa.png)


## ShiftEncoding

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
    initialize=Randomly(),
    size=image_size,
    default_encoding=StandardEncoding(),
    layers=Layer(
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
    initialize=ByImage(
        image="init_image.png",
        strength=0.8,
    ),
    size=image_size,
    layers=[
      Layer(
        prompt="black dog",
        negative_prompt="white cat",
        cfg_scale=7.5,
        mask_by="mask_image_1.png",
      ),
      Layer(
        prompt="white cat",
        negative_prompt="black dog",
        cfg_scale=7.5,
        mask_by="mask_image_2.png",
      ),
    ]
)[0]
display(image)
```

This usually works, but sometimes different layers may affect each other especially when the prompts of the layers are similar. In such cases, we can specify layers as `distinct` to avoid interference. Here is an example.

```python
image_size = (512, 512)  # width, height
image = pipe(
    num_steps=30,
    initialize=ByImage(
        image="init_image.png",
        strength=0.8,
    ),
    size=image_size,
    layers=[
      Layer(
        prompt="white cat",
        negative_prompt="black dog",
        cfg_scale=7.5,
        mask_by="mask_image_1.png",
        is_distinct=True,
      ),
      Layer(
        prompt="white cat",
        negative_prompt="black dog",
        cfg_scale=7.5,
        mask_by="mask_image_2.png",
        is_distinct=True,
      ),
    ]
)[0]
display(image)
```

### Text to image with multiple different prompts

Layers also enables us to use different prompts for different part of the image, such as background and foreground. This is conceptually similar to inpainting, but instead of using a provided image as a background, it also uses text to image for generating the background image. Here is an example.

```python
image_size = (512, 512)  # width, height
image = pipe(
    num_steps=30,
    initialize=Randomly(),
    size=image_size,
    layers=[
      Layer(
        prompt="grass field, blue sky",
        negative_prompt="tree",
        cfg_scale=7.5,
      ),
      Layer(
        prompt="white cat",
        negative_prompt="black dog",
        cfg_scale=7.5,
        mask_by="mask_image.png",
      ),
    ]
)[0]
display(image)
```

Similar to inpainting above, it is also possible to use 2+ layers for the foreground objects, and `is_distinct=True` is available as well. You should apply `is_distinct=True` only to the foreground layers in this case.

### Image to image with multiple different strengths

In image to image method, `strength` defines how much we expect the original image to be changed in the final output. The strength value applies to the entire image. However we may want to apply different strengths to different parts of the image, such as background and foreground. Layers offer the ability for that.

When you want to apply lower strength to the background, you can set up the pipeline as follows.

```python
image_size = (512, 512)  # width, height
image = pipe(
    num_steps=30,
    initialize=ByImage(
        image="init_image.png",
        strength=0.8,
    ),
    size=image_size,
    layers=[
      Layer(
        prompt="grass field, blue sky",
        negative_prompt="tree",
        cfg_scale=7.5,
        skip_until=0.3,
      ),
      Layer(
        prompt="white cat",
        negative_prompt="black dog",
        cfg_scale=7.5,
        mask_by="mask_image.png",
      ),
    ]
)[0]
display(image)
```

This example applies the strength 0.3 to the background (green field, blue sky) while applying the strength 0.8 to the foreground (white cat).
