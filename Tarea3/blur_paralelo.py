import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from scipy.misc import imread, imsave

TPB = 16

mod = SourceModule("""
    #include <stdio.h>
    #define CHANNELS 3
    #define BLUR_SIZE 5

    __global__ void blur(float *image, float *image_blur, int *dims) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int width = dims[0];
        int height = dims[1];

        if (col < width && row < height) {
            int indexI3 = (row * width + col)*CHANNELS;
            int pixValR = 0;
            int pixValG = 0;
            int pixValB = 0;
            int pixels = 0;

            for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) {
                for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
                    int curRow = row + blurRow;
                    int curCol = col + blurCol;
                    int indexB3 = (curRow * width + curCol)*CHANNELS;
                    
                    if(curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
                        pixValR += image[indexB3];
                        pixValG += image[indexB3+1];
                        pixValB += image[indexB3+2];
                        pixels++;
                    }
                }
            }
            image_blur[indexI3] = pixValR/pixels;
            image_blur[indexI3+1] = pixValG/pixels;
            image_blur[indexI3+2] = pixValB/pixels;
        }
    }
""")

def product(vector):
    p = 1
    for vi in vector:
        p *= vi
    return p

def parallel(input_name="test.jpg", output_name="test_blur.jpg"):
    # get kernel
    blur_kernel = mod.get_function("blur")    
    # get image    
    image = imread(input_name).astype(np.float32)
    # get the dimensions of the image
    width, height, channels = image.shape
    dims = np.array([width, height]).astype(np.int32)
    # change dimension to linear
    image = image.reshape(product(image.shape))
    image_blur = np.zeros_like(image).reshape(product(image.shape))
    # convert image to grayscale
    blur_kernel(cuda.In(image), cuda.InOut(image_blur), cuda.In(dims),
        block=(TPB, TPB, 1), grid=( int((width+TPB)//TPB), int((height+TPB)//TPB), 1) )
    # save image
    imsave(output_name, image_blur.reshape(width, height, channels))

if __name__ == '__main__':
    parallel()
