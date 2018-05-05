import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from scipy.misc import imread, imsave

TPB = 16

mod = SourceModule("""
    #include <stdio.h>
    
    __global__ void two_x(float *image, float *upscaling, int *dims) {
        const int CHANNELS = 3;
        const int SCALE = 2;
        const int ix = threadIdx.x + blockIdx.x * blockDim.x; // index i in upscaling
        const int iy = threadIdx.y + blockIdx.y * blockDim.y; // index j in upscaling  
        
        if(ix < dims[0]*SCALE && iy < dims[1]*SCALE){
            const int ix2 = ix/2; // index i in image
            const int iy2 = iy/2; // index j in image
            const int index_image = (iy2 + ix2 * dims[1]) * CHANNELS;
            const int index_upscaling = (iy + ix*dims[1]*SCALE) * CHANNELS;
            // RGB
            for(int rgb=0 ; rgb < CHANNELS ; rgb++){
                upscaling[index_upscaling+rgb] = image[index_image+rgb];
            }   
        }   
    }

""")

def scale_x(image, upscaling, image_width, image_height, scale=2):
    """
    Unicamente funciona para una escala de 2
    """
    for ix in range(image_width):
        for iy in range(image_height):
            """
            Suponiendo que estos son los pixeles
            el pixel 1 es el conocido, 2,3 y 4 son
            los pixeles que se igualaran al pixel 1.    
             _ _ _ _
            | 1 | 2 |
            |_ _|_ _|
            | 3 | 4 |
            |_ _|_ _|
            
            """
            index_image = (iy + ix * image_height) * 3
            index_upscaling = (iy*scale + scale * scale * ix * image_height) * 3
            # indices donde se replicara el pixel
            index_upscaling = [ index_upscaling+3*i+j*scale*3*image_height for j in range(scale) for i in range(scale) ]
            
            for index in index_upscaling:
                # pixeles
                upscaling[index] = image[ index_image ]
                upscaling[index+1] = image[ index_image+1 ]
                upscaling[index+2] = image[ index_image+2 ]
            

def serial(input_name="test.jpg", output_name="test_upscaling_serial.jpg"):
    # get image    
    image = imread(input_name)
    # get the dimensions of the image
    image_width, image_height, image_channels = image.shape
    # change dimension to linear
    image = image.reshape(image_width*image_height*image_channels)
    upscaling = np.zeros(shape=2*image_width*2*image_height*image_channels)
    # inicialize upscaling
    scale_x(image, upscaling, image_width, image_height)
    # save image
    imsave(output_name, upscaling.reshape((2*image_width,2*image_height,image_channels)))

def parallel(input_name="test.jpg", output_name="test_upscaling_parallel.jpg"):
    # get kernel
    two_x_kernel = mod.get_function("two_x")    
    # get image    
    image = imread(input_name).astype(np.float32)
    # get the dimensions of the image
    width, height, channels = image.shape
    dims = np.array([width, height]).astype(np.int32)
    # change dimension to linear
    image = image.reshape(width * height * channels)
    upscaling = np.zeros(shape=2*width*2*height*channels)
    # convert image to image 2x (upscaling)
    two_x_kernel(cuda.In(image), cuda.InOut(upscaling), cuda.In(dims),
        block=(TPB, TPB, 1), grid=( (2*width+TPB)//TPB, (2*height+TPB)//TPB, 1) )
    # save image
    imsave(output_name, upscaling.reshape((2*width, 2*height, channels)))

if __name__ == '__main__':
    #serial()
    parallel()
