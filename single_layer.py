import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import math
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

# Importing or Downloading the Inception model
import inception5h
inception5h.maybe_download()
model = inception5h.Inception5h()
len(model.layer_tensors)


def load_image(filename):
    image = PIL.Image.open(filename)
    return np.float32(image)

def save_image(image, filename):
    image = np.clip(image, 0.0, 255.0)
    image = image.astype(np.uint8)
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')# 'jpeg' Try png for better quality

def plot_image(image):
    if False:
        image = np.clip(image / 255.0, 0.0, 1.0)
        plt.imshow(image, interpolation='lanczos')
        #plt.show()
    else:
        image = np.clip(image, 0.0, 255.0)
        image = image.astype(np.uint8)

def normalize_image(x):
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm

def plot_gradient(gradient):
    gradient_normalized = normalize_image(gradient)
    plt.imshow(gradient_normalized, interpolation='bilinear')
    #plt.show()

def resize_image(image, size=None, factor=None):
    if factor is not None:
        size = np.array(image.shape[0:2]) * factor
        size = size.astype(int)
    else:
        size = size[0:2]
    size = tuple(reversed(size))
    img = np.clip(image, 0.0, 255.0)
    img = img.astype(np.uint8)
    img = PIL.Image.fromarray(img)
    img_resized = img.resize(size, PIL.Image.LANCZOS)
    img_resized = np.float32(img_resized)
    return img_resized

def get_tile_size(num_pixels, tile_size=400):
    num_tiles = int(round(num_pixels / tile_size))
    num_tiles = max(1, num_tiles)
    actual_tile_size = math.ceil(num_pixels / num_tiles)
    return actual_tile_size

def tiled_gradient(gradient, image, tile_size=400):
    grad = np.zeros_like(image)
    x_max, y_max, _ = image.shape
    x_tile_size = get_tile_size(num_pixels=x_max, tile_size=tile_size)
    x_tile_size4 = x_tile_size // 4
    y_tile_size = get_tile_size(num_pixels=y_max, tile_size=tile_size)
    y_tile_size4 = y_tile_size // 4
    x_start = random.randint(-3 * x_tile_size4, -x_tile_size4)
    while x_start < x_max:
        x_end = x_start + x_tile_size
        x_start_lim = max(x_start, 0)
        x_end_lim = min(x_end, x_max)
        y_start = random.randint(-3 * y_tile_size4, -y_tile_size4)
        while y_start < y_max:
            y_end = y_start + y_tile_size
            y_start_lim = max(y_start, 0)
            y_end_lim = min(y_end, y_max)
            img_tile = image[x_start_lim:x_end_lim,
                       y_start_lim:y_end_lim, :]
            feed_dict = model.create_feed_dict(image=img_tile)
            g = session.run(gradient, feed_dict=feed_dict)
            g /= (np.std(g) + 1e-8)
            grad[x_start_lim:x_end_lim,
            y_start_lim:y_end_lim, :] = g
            y_start = y_end
        x_start = x_end
    return grad


# ============================>> DREAM MANIPULATION SECTION <<===========================================
def optimize_image(layer_tensor, image,
                   num_iterations=10, step_size=3.0, tile_size=400,
                   show_gradient=False):

    img = image.copy()
    print("Processing image:", end="")
    gradient = model.get_gradient(layer_tensor)

    for i in range(num_iterations):
        grad = tiled_gradient(gradient=gradient, image=img)
        sigma = (i * 4.0) / num_iterations + 0.5
        grad_smooth1 = gaussian_filter(grad, sigma=sigma)
        grad_smooth2 = gaussian_filter(grad, sigma=sigma * 2)
        grad_smooth3 = gaussian_filter(grad, sigma=sigma * 0.5)
        grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)
        step_size_scaled = step_size / (np.std(grad) + 1e-8)
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

    print()
    return img


# ============================>> MAIN RUN OPTIONS <<=========================================
def recursive_optimize(layer_tensor, image,
                       num_repeats=4, rescale_factor=0.7, blend=0.2,
                       num_iterations=10, step_size=3.0,
                       tile_size=400):


    if num_repeats > 0:
        sigma = 0.5
        img_blur = gaussian_filter(image, sigma=(sigma, sigma, 0.0))
        img_downscaled = resize_image(image=img_blur,
                                      factor=rescale_factor)

        img_result = recursive_optimize(layer_tensor=layer_tensor,
                                        image=img_downscaled,
                                        num_repeats=num_repeats - 1,
                                        rescale_factor=rescale_factor,
                                        blend=blend,
                                        num_iterations=num_iterations,
                                        step_size=step_size,
                                        tile_size=tile_size)

        img_upscaled = resize_image(image=img_result, size=image.shape)
        image = blend * image + (1.0 - blend) * img_upscaled
    print("\nLayer:", layer_tensor)# This needs to be modified to describe the tensor layer correctly
    print("Recursive level:", num_repeats)
    print("Iterations:", num_iterations)


    # Process the image using the DeepDream algorithm.
    img_result = optimize_image(layer_tensor=layer_tensor,
                                image=image,
                                num_iterations=num_iterations,
                                step_size=step_size,
                                tile_size=tile_size)
    return img_result


session = tf.InteractiveSession(graph=model.graph)
image = load_image(filename='../../images/4.jpg')#==================================>> LOAD IMAGE HERE <<========

### ===========>> DEEP-DREAM MODEL LAYERS <<=================
# Layer[0][:,:,:,:]:     conv2d0:0           Renders: Greyscale Patterns
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

# Layer[ ][:,:,:,0:1]:   strided_slice:0     Renders: ???
# Layer[ ][:,:,:,0:1]:  strided_slice_1:0    Renders: ???
# Layer[ ][:,:,:,0:1]:  strided_slice_2:0    Renders: ???
# Layer[ ][:,:,:,0:1]:  strided_slice_3:0    Renders: ???
# Layer[ ][:,:,:,0:1]:  strided_slice_4:0    Renders: ???
# Layer[ ][:,:,:,0:1]:  strided_slice_5:0    Renders: ???
# Layer[ ][:,:,:,0:1]:  strided_slice_6:0    Renders: ???


# [:,:,:,0:1]
layer_tensor = model.layer_tensors[4]#[:,:,:,0:3]

img_result = optimize_image(layer_tensor, image,
                   num_iterations=30, step_size=6.0, tile_size=400,
                   show_gradient=True)

img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=80, step_size=3.0, rescale_factor=0.99,
                 num_repeats=7, blend=0.2)

# step_size=3.0
# rescale_factor=0.7
# blend=0.2

dd_img_path = '../../images/deep-dream-images/single_layer/'
save_image(img_result, filename=dd_img_path+'dd_layer_4.png')#=====>> SAVE IMAGE HERE <<======

play_finished()

