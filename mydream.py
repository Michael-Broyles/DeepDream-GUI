import matplotlib.pyplot as plt
import tensorflow as tf
#print(tf.__version__)# Prints our version of TensorFlow
import numpy as np
import random
import math
# Image manipulation.
import PIL.Image
from scipy.ndimage.filters import gaussian_filter
# Music that plays when the dream is finished
from pygame import mixer, time
def play_finished():
    file = '../../audio/power_up.ogg'
    mixer.init()
    mixer.music.load(file)
    mixer.music.play(loops=0, start=0.0)# Plays the song after the iname is saved to disk
    while mixer.music.get_busy():# Keeps the song playing
        time.Clock().tick(10)# Parameter is miliseconds

import inception5h
# The Inception 5h model is downloaded from the internet.
# This is the default directory where you want to save the data-files.
# The directory will be created if it does not exist.
# inception.data_dir = 'inception/5h/'
# Download the data for the Inception model if it doesn't already exist in the directory. It is 50 MB.
inception5h.maybe_download()

# Load the Inception model so it is ready to be used.
model = inception5h.Inception5h()
len(model.layer_tensors)


# This function loads an image and returns it as a numpy array of floating-points.
def load_image(filename):
    image = PIL.Image.open(filename)

    return np.float32(image)


# Save an image as a jpeg-file.
# The image is given as a numpy array with pixel-values between 0 and 255.
def save_image(image, filename):
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)

    # Convert to bytes.
    image = image.astype(np.uint8)

    # Write the image-file in jpeg-format.
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')# 'jpeg' Try png for better quality


# This function plots an image.
# Using matplotlib gives low-resolution images.
# Using PIL gives pretty pictures.
def plot_image(image):
    # Assume the pixel-values are scaled between 0 and 255.

    if False:
        # Convert the pixel-values to the range between 0.0 and 1.0
        image = np.clip(image / 255.0, 0.0, 1.0)

        # Plot using matplotlib.
        plt.imshow(image, interpolation='lanczos')
        plt.show()
    else:
        # Ensure the pixel-values are between 0 and 255.
        image = np.clip(image, 0.0, 255.0)

        # Convert pixels to bytes.
        image = image.astype(np.uint8)

        # Convert to a PIL-image and display it.
        #display(PIL.Image.fromarray(image))

# Normalize an image so its values are between 0.0 and 1.0. This is useful for plotting the gradient.
def normalize_image(x):
    # Get the min and max values for all pixels in the input.
    x_min = x.min()
    x_max = x.max()

    # Normalize so all values are between 0.0 and 1.0
    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm


# This function plots the gradient after normalizing it.
def plot_gradient(gradient):
    # Normalize the gradient so it is between 0.0 and 1.0
    gradient_normalized = normalize_image(gradient)

    # Plot the normalized gradient.
    plt.imshow(gradient_normalized, interpolation='bilinear')
    #plt.show()


####
####
def resize_image(image, size=None, factor=None):
    # If a rescaling-factor is provided then use it.
    if factor is not None:
        # Scale the numpy array's shape for height and width.
        size = np.array(image.shape[0:2]) * factor

        # The size is floating-point because it was scaled.
        # PIL requires the size to be integers.
        size = size.astype(int)
    else:
        # Ensure the size has length 2.
        size = size[0:2]

    # The height and width is reversed in numpy vs. PIL.
    size = tuple(reversed(size))

    # Ensure the pixel-values are between 0 and 255.
    img = np.clip(image, 0.0, 255.0)

    # Convert the pixels to 8-bit bytes.
    img = img.astype(np.uint8)

    # Create PIL-object from numpy array.
    img = PIL.Image.fromarray(img)

    # Resize the image.
    img_resized = img.resize(size, PIL.Image.LANCZOS)

    # Convert 8-bit pixel values back to floating-point.
    img_resized = np.float32(img_resized)

    return img_resized

# This is a helper-function for determining an appropriate tile-size.
# The desired tile-size is e.g. 400x400 pixels, but the actual tile-size will depend on the image-dimensions.
def get_tile_size(num_pixels, tile_size=400):
    """
    num_pixels is the number of pixels in a dimension of the image.
    tile_size is the desired tile-size.
    """

    # How many times can we repeat a tile of the desired size.
    num_tiles = int(round(num_pixels / tile_size))

    # Ensure that there is at least 1 tile.
    num_tiles = max(1, num_tiles)

    # The actual tile-size.
    actual_tile_size = math.ceil(num_pixels / num_tiles)

    return actual_tile_size

# This helper-function computes the gradient for an input image.
# The image is split into tiles and the gradient is calculated for each tile.
# The tiles are chosen randomly to avoid visible seams / lines in the final DeepDream image.
def tiled_gradient(gradient, image, tile_size=400):
    # Allocate an array for the gradient of the entire image.
    grad = np.zeros_like(image)

    # Number of pixels for the x- and y-axes.
    x_max, y_max, _ = image.shape

    # Tile-size for the x-axis.
    x_tile_size = get_tile_size(num_pixels=x_max, tile_size=tile_size)
    # 1/4 of the tile-size.
    x_tile_size4 = x_tile_size // 4

    # Tile-size for the y-axis.
    y_tile_size = get_tile_size(num_pixels=y_max, tile_size=tile_size)
    # 1/4 of the tile-size
    y_tile_size4 = y_tile_size // 4

    # Random start-position for the tiles on the x-axis.
    # The random value is between -3/4 and -1/4 of the tile-size.
    # This is so the border-tiles are at least 1/4 of the tile-size,
    # otherwise the tiles may be too small which creates noisy gradients.
    x_start = random.randint(-3 * x_tile_size4, -x_tile_size4)

    while x_start < x_max:
        # End-position for the current tile.
        x_end = x_start + x_tile_size

        # Ensure the tile's start- and end-positions are valid.
        x_start_lim = max(x_start, 0)
        x_end_lim = min(x_end, x_max)

        # Random start-position for the tiles on the y-axis.
        # The random value is between -3/4 and -1/4 of the tile-size.
        y_start = random.randint(-3 * y_tile_size4, -y_tile_size4)

        while y_start < y_max:
            # End-position for the current tile.
            y_end = y_start + y_tile_size

            # Ensure the tile's start- and end-positions are valid.
            y_start_lim = max(y_start, 0)
            y_end_lim = min(y_end, y_max)

            # Get the image-tile.
            img_tile = image[x_start_lim:x_end_lim,
                       y_start_lim:y_end_lim, :]

            # Create a feed-dict with the image-tile.
            feed_dict = model.create_feed_dict(image=img_tile)

            # Use TensorFlow to calculate the gradient-value.
            g = session.run(gradient, feed_dict=feed_dict)

            # Normalize the gradient for the tile. This is
            # necessary because the tiles may have very different
            # values. Normalizing gives a more coherent gradient.
            g /= (np.std(g) + 1e-8)

            # Store the tile's gradient at the appropriate location.
            grad[x_start_lim:x_end_lim,
            y_start_lim:y_end_lim, :] = g

            # Advance the start-position for the y-axis.
            y_start = y_end

        # Advance the start-position for the x-axis.
        x_start = x_end

    return grad

####
###### ==============>> DREAM MANIPULATION SECTION <<===========================================
####
def optimize_image(layer_tensor, image,
                   num_iterations=50, step_size=3.0, tile_size=400,
                   show_gradient=False):
    """
    Use gradient ascent to optimize an image so it maximizes the
    mean value of the given layer_tensor.

    Parameters:
    layer_tensor: Reference to a tensor that will be maximized.
    image: Input image used as the starting point.
    num_iterations: Number of optimization iterations to perform.
    step_size: Scale for each step of the gradient ascent.
    tile_size: Size of the tiles when calculating the gradient.
    show_gradient: Plot the gradient in each iteration.
    """

    # Copy the image so we don't overwrite the original image.
    img = image.copy()

    #print("Image before:")
    #plot_image(img)#**************************** TEST THIS OUT *******************

    print("Processing image:", end="")

    # Use TensorFlow to get the mathematical function for the
    # gradient of the given layer-tensor with regard to the
    # input image. This may cause TensorFlow to add the same
    # math-expressions to the graph each time this function is called.
    # It may use a lot of RAM and could be moved outside the function.
    gradient = model.get_gradient(layer_tensor)

    for i in range(num_iterations):
        # Calculate the value of the gradient.
        # This tells us how to change the image so as to
        # maximize the mean of the given layer-tensor.
        grad = tiled_gradient(gradient=gradient, image=img)

        # Blur the gradient with different amounts and add
        # them together. The blur amount is also increased
        # during the optimization. This was found to give
        # nice, smooth images. You can try and change the formulas.
        # The blur-amount is called sigma (0=no blur, 1=low blur, etc.)
        # We could call gaussian_filter(grad, sigma=(sigma, sigma, 0.0))
        # which would not blur the colour-channel. This tends to
        # give psychadelic / pastel colours in the resulting images.
        # When the colour-channel is also blurred the colours of the
        # input image are mostly retained in the output image.
        sigma = (i * 4.0) / num_iterations + 0.5
        grad_smooth1 = gaussian_filter(grad, sigma=sigma)
        grad_smooth2 = gaussian_filter(grad, sigma=sigma * 2)
        grad_smooth3 = gaussian_filter(grad, sigma=sigma * 0.5)
        grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)

        # Scale the step-size according to the gradient-values.
        # This may not be necessary because the tiled-gradient
        # is already normalized.
        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        # Update the image by following the gradient.
        img += grad * step_size_scaled

        if show_gradient:
            # Print statistics for the gradient.
            msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
            print(msg.format(grad.min(), grad.max(), step_size_scaled))

            # Plot the gradient.
            plot_gradient(grad)
        else:
            # Otherwise show a little progress-indicator.
            print(". ", end="")

    #print()
    #print("Image after:")
    plot_image(img)

    return img

# This helper-function downscales the input image several times and
# runs each downscaled version through the optimize_image() function above.
# This results in larger patterns in the final image.
# It also speeds up the computation.
# ========>> MAIN RUN OPTIONS <<==============
def recursive_optimize(layer_tensor, image,
                       num_repeats=6, rescale_factor=0.7, blend=0.2,
                       num_iterations=40, step_size=3.0,
                       tile_size=400):
    """
    Recursively blur and downscale the input image.
    Each downscaled image is run through the optimize_image()
    function to amplify the patterns that the Inception model sees.

    Parameters:
    image: Input image used as the starting point.
    rescale_factor: Downscaling factor for the image.
    num_repeats: Number of times to downscale the image.
    blend: Factor for blending the original and processed images.

    Parameters passed to optimize_image():
    layer_tensor: Reference to a tensor that will be maximized.
    num_iterations: Number of optimization iterations to perform.
    step_size: Scale for each step of the gradient ascent.
    tile_size: Size of the tiles when calculating the gradient.
    """

    # Do a recursive step?
    if num_repeats > 0:
        # Blur the input image to prevent artifacts when downscaling.
        # The blur amount is controlled by sigma. Note that the
        # colour-channel is not blurred as it would make the image gray.
        sigma = 0.5
        img_blur = gaussian_filter(image, sigma=(sigma, sigma, 0.0))

        # Downscale the image.
        img_downscaled = resize_image(image=img_blur,
                                      factor=rescale_factor)

        # Recursive call to this function.
        # Subtract one from num_repeats and use the downscaled image.
        img_result = recursive_optimize(layer_tensor=layer_tensor,
                                        image=img_downscaled,
                                        num_repeats=num_repeats - 1,
                                        rescale_factor=rescale_factor,
                                        blend=blend,
                                        num_iterations=num_iterations,
                                        step_size=step_size,
                                        tile_size=tile_size)

        # Upscale the resulting image back to its original size.
        img_upscaled = resize_image(image=img_result, size=image.shape)

        # Blend the original and processed images.
        image = blend * image + (1.0 - blend) * img_upscaled

    print("\nLayer:", layer_tensor)
    print("Recursive level:", num_repeats)
    print("Iterations:", num_iterations)


    # Process the image using the DeepDream algorithm.
    img_result = optimize_image(layer_tensor=layer_tensor,
                                image=image,
                                num_iterations=num_iterations,
                                step_size=step_size,
                                tile_size=tile_size)

    return img_result

# We need a TensorFlow session to execute the graph.
# This is an interactive session so we can continue adding gradient functions to the computational graph.
session = tf.InteractiveSession(graph=model.graph)

#
##
# NO IMAGES OVER 1000x1000 PIXLES WITHOUT A GPU AT THESE SETTINGS: REPEATS=5, ITERATIONS=30
# ../../../../Pictures/paola.PNG
##
#
image = load_image(filename='../../images/mountain.jpg')#==================================>> LOAD IMAGE HERE <<========
#plot_image(image)

### ===========>> DEEP-DREAM MODEL LAYERS <<=================
##
#  Layer 1: Wavy
#  Layer 2: Lines
#  Layer 3: Boxes
#  Layer 4: circles???
#  Layer 5: eyes
#  Layer 6: dogs, bears, cute animals
#  Layer 7: faces, buildings
#  Layer 8: fish, frogs, reptilian eyes
#  Layer 9: Snakes
#  Layer 10: monkies, kizards, snakes, ducks
#  Layer 11: Patterns???
##

# Layer: conv2d1:0
layer_tensor = model.layer_tensors[1]#[:,:,:,0:3]
# Now run the DeepDream optimization algorithm for 10 iterations with a step-size of 6.0,
# which is twice as high as in the recursive optimizations below.
# We also show the gradient for each iteration and
# you should note the visible artifacts in the seams between the tiles.
img_result = optimize_image(layer_tensor, image,
                   num_iterations=50, step_size=6.0, tile_size=400,
                   show_gradient=True)

img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
              num_iterations=50, step_size=3.0, rescale_factor=0.7,
              num_repeats=7, blend=0.2)

# Layer: conv2d2:0
layer_tensor = model.layer_tensors[2]#[:,:,:,0:1]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=50, step_size=3.0, rescale_factor=0.7,
                 num_repeats=7, blend=0.2)

# Layer: mixed3a:0
layer_tensor = model.layer_tensors[3]#[:,:,:,0:1]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
              num_iterations=50, step_size=3.0, rescale_factor=0.7,
              num_repeats=7, blend=0.2)

# Layer: mixed3b:0
layer_tensor = model.layer_tensors[4]#[:,:,:,0:1]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=50, step_size=3.0, rescale_factor=0.7,
                 num_repeats=7, blend=0.2)

# Layer: mixed4a:0
layer_tensor = model.layer_tensors[5]#[:,:,:,0:1]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=50, step_size=3.0, rescale_factor=0.7,
                 num_repeats=7, blend=0.2)

# Layer: mixed4b:0
layer_tensor = model.layer_tensors[6]#[:,:,:,0:1]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=50, step_size=3.0, rescale_factor=0.7,
                 num_repeats=7, blend=0.2)

# This is an example of maximizing only a subset of a layer's feature-channels using the DeepDream algorithm.
# In this case it is the layer with index 7 and only its first 3 feature-channels that are maximized.
# Layer: strided_slice:0
layer_tensor = model.layer_tensors[7]#[:,:,:,0:3]#[:,:,:,0:3]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=60, step_size=3.0, rescale_factor=0.7,
                 num_repeats=7, blend=0.2)

# Layer: mixed4d:0
layer_tensor = model.layer_tensors[8]#[:,:,:,0:3]#[:,:,:,0:1]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=50, step_size=6.0, rescale_factor=0.7,# step size 3
                 num_repeats=7, blend=0.2)

# Layer: mixed4e:0
layer_tensor = model.layer_tensors[9]#[:,:,:,0:3]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=50, step_size=3.0, rescale_factor=0.7,
                 num_repeats=7, blend=0.2)

# Layer: mixed5a:0
layer_tensor = model.layer_tensors[10]#[:,:,:,0:3]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=60, step_size=3.0, rescale_factor=0.7,
                 num_repeats=6, blend=0.2)

# This example shows the result of maximizing the first feature-channel of the final layer in the Inception model.
# It is unclear what patterns this layer and feature-channel might be recognizing in the input image.
# Layer: strided_slice_1:0
layer_tensor = model.layer_tensors[11]#[:,:,:,0]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=60, step_size=3.0, rescale_factor=0.7,
                 num_repeats=6, blend=0.2)

# 'num_iterations' will increase the effects in that layer, Max: 99 ???
# 'num_repeats'effects how clear the dream will be



# png, jpeg, jpg ALL WORK FINE
save_image(img_result, filename='../../images/deep-dream-images/deepdream_mountain_2.png')#===>> SAVE IMAGE HERE <<====


play_finished()

# Layer[0][:,:,:,:]:     conv2d0:0           Renders: Greyscale Patterns (Not manipulated in this script)
# Layer[1][:,:,:,:]:     conv2d1:0           Renders: Wavy
# Layer[2][:,:,:,:]:     conv2d2:0           Renders: Lines
# Layer[3][:,:,:,:]:     mixed3a:0           Renders: Boxes
# Layer[4][:,:,:,:]:     mixed3b:0           Renders: Circles
# Layer[5][:,:,:,:]:     mixed4a:0           Renders: Eyes
# Layer[6][:,:,:,:]:     mixed4b:0           Renders: Dogs, Bears, Cute Animals
# Layer[7][:,:,:,:]:     mixed4c:0           Renders: Faces, Buildings
# Layer[8][:,:,:,:]:     mixed4d:0           Renders: Fish, Frogs, Reptilian Eyes
# Layer[9][:,:,:,:]:     mixed4e:0           Renders: Snake Heads
# Layer[10][:,:,:,:]:    mixed5a:0           Renders: Monkies, Lizards, Ducks
# Layer[11][:,:,:,:]:    mixed5b:0           Renders: Fish, Lizards, Birds

# Layer[ ][:,:,:,0:1]:  strided_slice:0      Renders: ???
# Layer[ ][:,:,:,0:1]:  strided_slice_1:0    Renders: ???
# Layer[ ][:,:,:,0:1]:  strided_slice_2:0    Renders: ???
# Layer[ ][:,:,:,0:1]:  strided_slice_3:0    Renders: ???
# Layer[ ][:,:,:,0:1]:  strided_slice_4:0    Renders: ???
# Layer[ ][:,:,:,0:1]:  strided_slice_5:0    Renders: ???
# Layer[ ][:,:,:,0:1]:  strided_slice_6:0    Renders: ???


