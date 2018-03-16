import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from scipy.misc import imread, imsave

TPB = 16

mod = SourceModule("""
    #include <stdio.h>
    #define CHANNELS 3
    
    __global__ void greyscale(float *image, int *dims) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        
        if (i < dims[0] && j < dims[1]) {
            int index = i + j*dims[0]; 
            int index3 = index*CHANNELS;   
            float value = 0.21*image[index3] + 0.71*image[index3+1] + 0.07*image[index3+2];
            image[index3] = value;
            image[index3+1] = value;
            image[index3+2] = value;
        }
    }

""")

def product(vector):
    p = 1
    for vi in vector:
        p *= vi
    return p

def parallel(input_name="test.jpg", output_name="test_grey.jpg"):
    # get kernel
    greyscale_kernel = mod.get_function("greyscale")    
    # get image    
    image = imread(input_name).astype(np.float32)
    # get the dimensions of the image
    width, height, channels = image.shape
    dims = np.array([width, height]).astype(np.int32)
    # change dimension to linear
    image = image.reshape(product(image.shape))
    # convert image to grayscale
    greyscale_kernel(cuda.InOut(image), cuda.In(dims),
        block=(TPB, TPB, 1), grid=( int((width+TPB)//TPB), int((height+TPB)//TPB), 1) )
    # save image
    imsave(output_name, image.reshape(width, height, channels))

if __name__ == '__main__':
    parallel()