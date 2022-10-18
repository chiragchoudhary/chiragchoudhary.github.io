---
title: 'A walk-through on Neural Style Transfer'
date: 2022-10-09
permalink: /posts/neural_style_transfer/
tags:
  - Deep Learning
  - NST
  - Computer Vision
---
<style>
p > img {
height: 150px;
margin-bottom:5px;
}
</style>
In the very first post on this blog, we look at application of deep learning to the area of style transfer, popularly known as Neural Style Transfer.
This tutorial assumes basic knowledge of CNNs and the Tensorflow/Keras python libraries.
    
## Introduction
The aim of Neural Style Transfer (NST) is to embed the style of an image (aka style image) onto a base image (aka content image). It was proposed by Leon A. Gatys et al. in their 2015 paper *[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)*. The idea is similar to CycleGANs, which also transposes images between two style domains. However, in NST, we don't have a training set, but instead we work with just two images. In terms of training, it means that we optimize the combined image rather than optimizing the weights of an ML model.

<p align="center">
<img src="{{site.url}}/images/nst/starry_night_resize.jpg" style="height: 150">
<img src="{{site.url}}/images/plus.png" style="height: 30px">
<img src="{{ site.url }}/images/nst/big_ben.jpg" style="height:150">
<img src="{{site.url}}/images/eq.png" style="height: 30px">
<img src="{{ site.url }}/images/nst/big_ben_and_starry_night.png" style="height:150; margin-left: 0px">
</p>

The transformed image needs to adopt the style of the style image while conserving the contents of the content image. To achieve this style transform, the algorithm represents these two properties mathematically via loss functions:

### Content Loss
The various activations maps of a ConvNet capture different information about the input image. The activation maps in earlier layers capture more local information about the image, such as edges, colors, etc., whereas the activation maps in deeper layers capture more global/abstract information about the input, such as the class of the object. This means that two images which have very similar contents should also have similar activation values for the upper layers. For an input image, the content is defined as the activation values for a deeper layer of a pre-trained ConvNet. The content loss is then an L2 norm between the activations of that layer, computed over the final output, and the activations of the same layer computed for our content image. The layer chosen to represent the contents of an image is a hyper-parameter of our algorithm. We use a pre-trained VGG-19 for our activations.

### Style Loss

Consider the style image show above. We can make some reasonable remarks about the style of the image:


The intuition behind ConvNets is that different activation maps of a given layer capture different features at the same spatial level in the input image. For instance, one activation map has high activations for areas of image which have green color, whereas another activation map could activate when it detects spikes. Both the activation maps combined allow the layer to detect grass, a more abstract feature than both green or spikes.

The original papers suggests that the style of an image can be represented by the correlations between different activation maps across multiple layers. For each layer, we compute a matrix of correlation of a pair of activation maps in that layer. This matrix is known as a *gram matrix*. The style loss for a layer is then defined as the L2 norm of the gram matrix of activations for the style image, and the gram matrix for the same layer for the combined image. The overall style loss is the normalized sum of style losses for multiple layers.

Intuitively, based on the above example of style image, the algorithm will add yellow colored rings to spherical areas of the content image, and grassy texture to dark areas of the content image. Similarly, it will add bright blue lines (texture) to the blue regions of the content image. This ensures that the activation maps which activate on blue color and blue line texture are highly correlated for both the style image as well as the combined image.


### Total Variance Loss
The total variance loss represents the noise in our combined image output. This loss allows us to tune the amount of smoothness in the output. To measure noise, we shift the image one pixel down and calculate the sum of squared difference (SSD) between the shifted and original image. We repeat the same procedure, but shift the image one pixel to the right. Note that when we compute the SSD, we compute it over the overlapping patches (size (M-1)x(N-1)) between the shifted and original image.

The algorithm optimizes the total loss, which is a weighted sum of the three losses. By varying the relative weights, we can modify the style of the final combined image.
## Implementation
Note: A full implementation of the project is available [here](https://github.com/choudharyc/Neural-Style-Transfer).

### Directory Structure
The python script assumes the following directory structure:
   
    .
    ├── data
    │    ├── output
    │    └── src
    │        ├── content
    │        └── style
    ├── main.py
    └── utils.py

### Import Libraries
```python
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import keras
from keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import load_img, img_to_array, save_img
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input

from utils import deprocess_image, save_animation
```

### Define Parameters
```python
content_loss_weight = 1e-5
style_loss_weight = 2.5e-3
variance_loss_weight = 1e-6

init_methods = ['content', 'style', 'noise']
reconstruction_types = ['content', 'style', 'both']

init_method = 'content'  # One of 'content', 'style' or 'noise'
reconstruction_type = 'both'  # One of 'content', 'style' or 'both'
epochs = 2000
num_epochs_per_save = 100
learning_rate = 2.0

style_feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
content_feature_layer = 'block2_conv2'
```

### Loading the content image and the style image
```python
content_image_path = Path('data') / 'src' / 'autumn_road.jpg'
style_image_path = Path('data') / 'src' / 'waves.jpg'
output_folder = Path('data') / 'output'

original_width, original_height = keras.utils.load_img(content_image_path).size
print(original_width, original_height)

img_height = 300
img_width = round(original_width * img_height / original_height)
print(img_width, img_height)

base_image = img_to_array(load_img(content_image_path, target_size=(img_height, img_width), interpolation='bicubic'))
style_image = img_to_array(load_img(style_image_path, target_size=(img_height, img_width), interpolation='bicubic'))

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(base_image / 255.0)
ax[0].axis('off')
ax[0].set_title('Base Image')
ax[1].imshow(style_image / 255.0)
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
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
```
### Initializing combined output tensor
```python
base_image_tensor = preprocess_input(np.expand_dims(base_image.copy(), axis=0))
style_image_tensor = preprocess_input(np.expand_dims(style_image.copy(), axis=0))

if init_method == 'content':
    combined_image_tensor = tf.Variable(base_image_tensor.copy())
elif init_method == 'noise':
    combined_image_tensor = tf.Variable(np.random.random(base_image_tensor.shape), dtype=np.float32)
else:
    combined_image_tensor = tf.Variable(style_image_tensor.copy())
```

### Training
```python
for epoch in range(epochs + 1):
    with tf.GradientTape() as tp:
        input_tensor = tf.concat([base_image_tensor, style_image_tensor, combined_image_tensor], 0)
        features = model(input_tensor)

        content_features_base_image = features[content_feature_layer][0]
        content_features_combined_image = features[content_feature_layer][2]

        content_loss = content_loss_weight * content_loss_fn(content_features_base_image, content_features_combined_image)

        style_loss = 0.0
        for layer in style_feature_layers:
            style_features_style_image = features[layer][1]
            style_features_combined_image = features[layer][2]
            channels = 3
            size = img_height * img_width
            style_loss_l = style_loss_fn(style_features_style_image, style_features_combined_image)
            style_loss_l = style_loss_l / (4.0 * (channels ** 2) * (size ** 2))
            style_loss += style_loss_l / (len(style_feature_layers))

        style_loss = style_loss_weight * style_loss

        variance_loss = variance_loss_fn(combined_image_tensor)
        variance_loss = variance_loss_weight * variance_loss

        content_losses.append(content_loss)
        style_losses.append(style_loss)
        variance_losses.append(variance_loss)

        if reconstruction_type == 'content':
            total_loss = content_loss
        elif reconstruction_type == 'style':
            total_loss = style_loss
        else:
            total_loss = content_loss + style_loss + variance_loss

        total_losses.append(total_loss.numpy())

    print(f"[Epochs: {epoch}/{epochs}] Total loss: {total_loss:.2f}, content loss: {content_loss:.2f}, style loss: {style_loss:.4f}, variance loss: {variance_loss:.2f}")
    grad = tp.gradient(total_loss, combined_image_tensor)
    optimizer.apply_gradients([(grad, combined_image_tensor)])

    # Apply Gradients
    if epoch % num_epochs_per_save == 0:
        # Save combined image
        combined_image = combined_image_tensor.numpy()[0]
        combined_image = deprocess_image(combined_image)
        fname = output_folder / f"{content_image_path.stem}_and_{style_image_path.stem}" / reconstruction_type / f'regenerate__{epoch:04}.png'
        fname.parent.mkdir(parents=True, exist_ok=True)
        save_img(fname, combined_image)
```

## Results

Here are some examples of three different styles applied to two different content images:

<p align="center" style="min-width:620px">
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
If we initialize the combined image to random noise and set the total loss to content loss only, the model simply reconstructs the content image. The final image reconstructed is the same, regardless of which layer we choose as our content feature layer.

The following animation shows the content reconstruction when using block2_conv2 VGG layer for content features:

<p align="center">
<img src="{{site.url}}/images/nst/autumn_road_and_waves_content_b2c2.gif" style="height: 250px">
</p>

### Style Reconstruction
Similarly, if we initialize the combined image to random noise and set the total loss to style loss only, the model generates an output which only has the stylist components of the style image, without any particular structure.

<p align="center">
<img src="{{site.url}}/images/nst/autumn_road_and_waves_style.gif" style="height: 250px">
</p>

### Tuning the style weight
By varying the style weight relative to the content weight, we can control how much style we want to add to the content image. Here, the image is initialized as the content image, the content weight is fixed at 1e-5, and the style weight values are incremented in multiples of 10 (1e-5, 1e-4, 1e-3, and 1e-2).

<p align="center">
<img src="{{site.url}}/images/nst/style_4.png" style="height: 200px">
<img src="{{site.url}}/images/nst/style_3.png" style="height: 200px">
<img src="{{site.url}}/images/nst/style_2.png" style="height: 200px">
<img src="{{site.url}}/images/nst/style_1.png" style="height: 200px">
</p>

<!--
### Tuning the total variance weight
Similarly, by changing the total variance weight, we can control how smooth we want the final image to look.
-->
### Hardware Specs
- CPU: 	AMD Ryzen 9 5900X 3.7 GHz 12-Core Processor
- GPU: NVIDIA Geforce RTX 3080 10GB

## References

1. [Deep Generative Learning book](https://learning.oreilly.com/library/view/generative-deep-learning/)
2. [The AI Epiphany NST video series](https://youtube.com/playlist?list=PLBoQnSflObcmbfshq9oNs41vODgXG-608) (highly recommended)
3. [Keras official NST example](https://keras.io/examples/generative/neural_style_transfer/)
