import numpy as np
from scipy.misc import imread, imsave

def product(vector):
    p = 1
    for vi in vector:
        p *= vi
    return p

def grayscale(image):
    vector = image.reshape(product(image.shape))

    for i in range( product(image.shape[:-1]) ):
        i3 = 3*i
        value = 0.21*vector[i3] + 0.71*vector[i3+1] + 0.07*vector[i3+2] 
        vector[i3], vector[i3+1], vector[i3+2] = value, value, value
        
    return vector

def serial(input_name='test.jpg', output_name='test_grey.jpg'):
    image = imread(input_name)
    # convert image to grayscale
    image_greyscale = grayscale(image).reshape(image.shape)
    # save image
    imsave(output_name, image_greyscale)

if __name__ == '__main__':
    serial()