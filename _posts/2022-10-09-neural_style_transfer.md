---
title: 'A walk-through on Neural Style Transfer'
date: 2022-10-09
permalink: /posts/neural_style_transfer/
tags:
  - Deep Learning
  - NST
---
<style>
img {
height: 150px
}
</style>
In the very first post on this blog, we look at application of deep learning to the area of style transfer, popularly known as Neural Style Transfer.
This tutorial assumes basic knowledge of CNNs and the Tensorflow/Keras python libraries.
    
## Introduction
The aim of Neural Style Transfer (NST) is to embed the style of an image (aka style image) onto a base image (aka content image). It was proposed by Leon A. Gatys et al. in their 2015 paper *A Neural Algorithm of Artistic Style*. The idea is similar to CycleGANs, which also transposes images between two style domains. However, in NST, we don't have a training set, but instead we work with just two images. In terms of training, it means that we optimize the combined image rather than optimizing the weights of an ML model.

<p align="center">
<img src="{{ site.url }}/images/nst/big_ben.jpg" style="height:200px">
<img src="{{ site.url }}/images/nst/big_ben_and_starry_night.png" style="height:200px; margin-left: 0px">
</p>

The transformed image needs to have the style of the style image, but also retain the contents of the content image. To achieve this style transform, the algorithm represents these two properties mathematically via loss functions:

### Content Loss

### Style Loss


### Total Variance Loss

The algorithm optimizes the total loss, which is a weighted sum of the three losses. By varying the relative weights, we can modify the style of the final combined image.
## Implementation
Note: A full implementation of the project is available [here](https://github.com/choudharyc/Neural-Style-Transfer).

### Directory Structure

### Import Libraries
```python
import os
from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sb

import tensorflow as tf
import keras
from keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import load_img, img_to_array, save_img
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
```

### Define Parameters
```python
content_loss_weight = 1e-5
style_loss_weight = 2.5e-3
variance_loss_weight = 1e-6

init_method = 'content' # One of 'content', 'style' or 'noise'
regenerate = 'content_and_style'

epochs = 4000
num_epochs_per_save = 100

img_height = 300

style_feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
content_feature_layer = 'block2_conv2'
```

### Loading the content image and the style image
```python
base_image_path = Path('data') / 'taj_mahal.jpg'
style_image_path = Path('data') / 'ben_giles.jpg'
tmp_data_path = Path('data') / 'tmp' /

original_width, original_height = keras.utils.load_img(base_image_path).size
print(original_width, original_height)

img_width = round(original_width*img_height/original_height)
base_image = img_to_array(load_img(base_image_path, target_size=(img_height, img_width), interpolation='bicubic'))
style_image = img_to_array(load_img(style_image_path, target_size=(img_height, img_width), interpolation='bicubic'))

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(base_image/255.0)
ax[0].axis('off')
ax[0].set_title('Base Image')
ax[1].imshow(style_image/255.0)
ax[1].axis('off')
ax[1].set_title('Style Image')

fig.show()
```

### Defining our feature extraction model
```python
vgg = VGG19(weights='imagenet', include_top=False)
output_dict = {layer.name: layer.output for layer in vgg.layers}
model = Model(inputs=vgg.input, outputs=output_dict)
```

### Loss Functions
```python
def content_loss_fn(base_features, combined_features):
    return tf.reduce_sum(tf.square(base_features - combined_features))

def variance_loss_fn(img):
    height, width = img.shape[1:3]
    a = tf.square(img[:, height-1, :width-1, :] - img[:, 1:, :width-1, :])
    b = tf.square(img[:, height-1, :width-1, :] - img[:, :height-1, 1:, :])
    
    return tf.reduce_sum(a+b)

def style_loss_fn(style_img, combined_img):
    def gram_matrix(feature_matrix):
        flattened_feature_matrix = K.batch_flatten(K.permute_dimensions(feature_matrix, (2, 0, 1)))
        gram = K.dot(flattened_feature_matrix, K.transpose(flattened_feature_matrix))
        return gram
    
    style_img_gram_matrix = gram_matrix(style_img)
    combined_img_gram_matrix = gram_matrix(combined_img)
    
    return tf.reduce_sum(tf.square(style_img_gram_matrix-combined_img_gram_matrix))
```

### Optimizer
```python
optimizer = tf.keras.optimizers.Adam(
                tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=2.0, decay_steps=500, decay_rate=0.96
                )
            )
```

### Training
```python
for epoch in range(4000, 6001):
    with tf.GradientTape() as tp:
        input_tensor = tf.concat([base_image_tensor, style_image_tensor, combined_image_tensor], 0)
        features = model(input_tensor)
                
        content_features_base_image = features['block3_conv2'][0]
        content_features_combined_image = features['block3_conv2'][2]
        
        content_loss = content_loss_weight * content_loss_fn(content_features_base_image, content_features_combined_image)
        
        style_loss = 0.0
        for layer in style_feature_layers:
            style_features_style_image = features[layer][1]
            style_features_combined_image = features[layer][2]
            height, width, channels = style_features_style_image.shape
            size = height * width
            style_loss_l = style_loss_fn(style_features_style_image, style_features_combined_image)
            style_loss_l = style_loss_l/(4.0 * (3**2) * ((img_height*img_width)**2))
            style_loss += style_loss_l/(len(style_feature_layers))
        
        style_loss = style_loss_weight * style_loss
        
        variance_loss = variance_loss_fn(combined_image_tensor)
        variance_loss = variance_loss_weight * variance_loss
        
        content_losses.append(content_loss)
        style_losses.append(style_loss)
        variance_losses.append(variance_loss)
        if regenerate == 'content':
            total_loss = content_loss
        elif regenerate == 'style':
            total_loss = style_loss
        elif regenerate == 'content_and_style':
            total_loss = content_loss + style_loss + variance_loss
        else:
            break
        total_losses.append(total_loss.numpy())
        
    print(f"[Epochs: {epoch}/{epochs}] Total loss: {total_loss:.2f}, content loss: {content_loss:.2f}, style loss: {style_loss:.4f}, variance loss: {variance_loss:.2f}")
    grad = tp.gradient(total_loss, combined_image_tensor)
    
    optimizer.apply_gradients([(grad, combined_image_tensor)])
    
    # Apply Gradients
    if epoch % num_epochs_per_save == 0:
        # Display combined image
        combined_image = combined_image_tensor.numpy()[0]
        combined_image = deprocess_image(combined_image)
        fname = tmp_data_path / regenerate / f'regenerate__{epoch:04}.png'
        save_img(fname, combined_image)

save_animation()
```

## Results

Here are some examples of three different styles applied to two different content images:

<p align="center">
<img src="{{site.url}}/images/nst/waves_resize.png" height=220px>
<img src="{{ site.url }}/images/nst/autumn_road_and_waves.png" height="250px" style="margin-left: 0px">
<img src="{{ site.url }}/images/nst/taj_mahal_and_waves.png" height="250px" style="margin-left: 0px">
<br>
<img src="{{ site.url }}/images/nst/starry_night_resize.jpg" height="250px">
<img src="{{ site.url }}/images/nst/autumn_road_and_starry_night.png" height="250px" style="margin-left: 0px">
<img src="{{ site.url }}/images/nst/taj_mahal_and_starry_night.png" height="250px" style="margin-left: 0px">
<br>
<img src="{{ site.url }}/images/nst/autumn_resize.jpg" height="250px">
<img src="{{ site.url }}/images/nst/autumn_road_and_autumn.png" height="250px" style="margin-left: 0px">
<img src="{{ site.url }}/images/nst/taj_mahal_and_autumn.png" height="250px" style="margin-left: 0px">
<br>
<img src="{{ site.url }}/images/nst/pink_flowers_resize.jpg" height="250px">
<img src="{{ site.url }}/images/nst/autumn_road_and_pink_flowers.png" height="250px" style="margin-left: 0px">
<img src="{{ site.url }}/images/nst/taj_mahal_and_pink_flowers.png" height="250px" style="margin-left: 0px">
</p>

### Content Reconstruction
If we set the total loss to content loss only, the model simply reconstructs the content image. The final image reconstructed is the same, regardless of which layer we choose as our content feature layer.

### Style Reconstruction


### Tuning the style weight
By varying the style weight relative to the content weight, we can control how much style we want to add to the content image.

### Tuning the total variance weight
Similarly, by changing the total variance weight, we can control how smooth we want the final image to look.

### Hardware Specs
- CPU: 	AMD Ryzen 9 5900X 3.7 GHz 12-Core Processor
- GPU: NVIDIA Geforce RTX 3080 10GB

## References

1. Deep Generative Learning book: https://learning.oreilly.com/library/view/generative-deep-learning/
2. The AI Epiphany NST video series (highly recommended): https://www.youtube.com/c/TheAiEpiphany
