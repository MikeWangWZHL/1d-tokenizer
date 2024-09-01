import demo_util
import numpy as np
import torch
from PIL import Image
import imagenet_classes
import os


torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)

config = demo_util.get_config("configs/titok_l32.yaml")

config.experiment.tokenizer_checkpoint = "checkpoints/tokenizer_titok_l32.bin"
config.experiment.generator_checkpoint = "checkpoints/generator_titok_l32.bin"

print(config)

titok_tokenizer = demo_util.get_titok_tokenizer(config)
print(titok_tokenizer)

import pdb; pdb.set_trace()

titok_generator = demo_util.get_titok_generator(config)
print(titok_generator)

import pdb; pdb.set_trace()

device = "cuda"
titok_tokenizer = titok_tokenizer.to(device)
titok_generator = titok_generator.to(device)

## Tokenize an Image into 32 discrete tokens

def tokenize_and_reconstruct(img_path):
    original_image = Image.open(img_path)
    image = torch.from_numpy(np.array(original_image).astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
    encoded_tokens = titok_tokenizer.encode(image.to(device))[1]["min_encoding_indices"]
    reconstructed_image = titok_tokenizer.decode_tokens(encoded_tokens)
    reconstructed_image = torch.clamp(reconstructed_image, 0.0, 1.0)
    reconstructed_image = (reconstructed_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
    reconstructed_image = Image.fromarray(reconstructed_image)
    print(f"Input Image is represented by codes {encoded_tokens} with shape {encoded_tokens.shape}")
    # print("orginal image:")
    # display(original_image)
    # print("reconstructed image:")
    # display(reconstructed_image)

    # save the images
    original_image.save("results/original_image.jpg")
    reconstructed_image.save("results/reconstructed_image.jpg")

    return original_image, reconstructed_image

original_image, reconstructed_image = tokenize_and_reconstruct("assets/ILSVRC2012_val_00010240.png")

import pdb; pdb.set_trace()

## class conditional generation
sample_labels = [torch.randint(0, 999, size=(1,)).item()]

# The guidance_scale and randomize_temperature can be adjusted to trade-off between quality and diversity.
generated_image = demo_util.sample_fn(
    generator=titok_generator,
    tokenizer=titok_tokenizer,
    labels=sample_labels,
    guidance_scale=3.5,
    randomize_temperature=1.0,
    num_sample_steps=8,
    device=device
)

for i in range(generated_image.shape[0]):
    print(f"labels {sample_labels[i]}, {imagenet_classes.imagenet_idx2classname[sample_labels[i]]}")
    # display(Image.fromarray(generated_image[i]))
    
    # save the images
    Image.fromarray(generated_image[i]).save(f"results/generated_image_{i}.jpg")