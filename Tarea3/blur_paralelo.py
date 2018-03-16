import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from scipy.misc import imread, imsave

TPB = 16

mod = SourceModule("""
    #include <stdio.h>
    #define CHANNELS 3
    #define BLUR_SIZE 2

    __global__ void blur(float *image, float *image_blur, int *mask, int *dims) {
        int row_i = blockIdx.x * blockDim.x + threadIdx.x;
        int col_i = blockIdx.y * blockDim.y + threadIdx.y;
        int image_width = dims[0];
        int image_height = dims[1];

        if (col_i < image_width && row_i < image_height) {
            int indexI3 = (row_i * image_width + col_i)*CHANNELS;
            int pixValR = 0;
            int pixValG = 0;
            int pixValB = 0;
            int pixels = 0;

            for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) {
                for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
                    int curRow = row_i + (blurRow*CHANNELS);
                    int curCol = col_i + (blurCol*CHANNELS);
                    int indexB3 = (curRow * image_width + curCol)*CHANNELS;
                    
                    if(curRow > -1 && curRow < image_height && curCol > -1 && curCol < image_width) {
                        pixValR += image[indexB3] * mask[(BLUR_SIZE-blurCol) + (BLUR_SIZE-blurRow)*(2*BLUR_SIZE+1)];
                        pixValG += image[indexB3+1] * mask[(BLUR_SIZE-blurCol) + (BLUR_SIZE-blurRow)*(2*BLUR_SIZE+1)];
                        pixValB += image[indexB3+2] * mask[(BLUR_SIZE-blurCol) + (BLUR_SIZE-blurRow)*(2*BLUR_SIZE+1)];
                        pixels += mask[(BLUR_SIZE-blurCol) + (BLUR_SIZE-blurRow)*(2*BLUR_SIZE+1)];
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

def parallel(input_name="test.jpeg", output_name="test_blur.jpeg"):
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
    mask = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]]) 
    mask = mask.reshape(product(mask.shape)).astype(np.int32)
    # convert image to grayscale
    blur_kernel(cuda.InOut(image), cuda.InOut(image_blur), cuda.InOut(mask), cuda.In(dims),
        block=(TPB, TPB, 1), grid=( int((width+TPB)//TPB), int((height+TPB)//TPB), 1) )
    # save image
    imsave(output_name, image_blur.reshape(width, height, channels))

if __name__ == '__main__':
    parallel()
