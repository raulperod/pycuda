import numpy as np
from scipy.misc import imread, imsave

def product(vector):
    p = 1
    for vi in vector:
        p *= vi
    return p

def blur(image):
    image_normal = image.reshape(product(image.shape))
    image_width, image_height, channels = image.shape
    image_blur = np.zeros_like(image).reshape(product(image.shape))
    mask = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])
    BLUR_SIZE = 2

    for col_i in range( image_width ):
        for row_i in range( image_height ):
            indexI3 = (row_i * image_width + col_i)*channels
            pixValR, pixValG, pixValB, pixels = (0,0,0,0)
            
            for blurRow in range(-BLUR_SIZE, BLUR_SIZE+1):
                for blurCol in range(-BLUR_SIZE, BLUR_SIZE+1):
                    curRow = row_i + (blurRow*channels)
                    curCol = col_i + (blurCol*channels)
                    indexB3 = (curRow * image_width + curCol)*channels
                    
                    if (image_height > curRow > -1) and (image_width > curCol > -1):
                        pixValR += image_normal[indexB3] * mask[BLUR_SIZE-blurRow][BLUR_SIZE-blurCol]
                        pixValG += image_normal[indexB3+1] * mask[BLUR_SIZE-blurRow][BLUR_SIZE-blurCol]
                        pixValB += image_normal[indexB3+2] * mask[BLUR_SIZE-blurRow][BLUR_SIZE-blurCol]
                        pixels += mask[BLUR_SIZE-blurRow][BLUR_SIZE-blurCol]
           
            image_blur[indexI3] = pixValR/pixels
            image_blur[indexI3+1] = pixValG/pixels
            image_blur[indexI3+2] = pixValB/pixels

    return image_blur
    

def serial(input_name='test.jpeg', output_name='test_blur.jpeg'):
    image = imread(input_name)
    # convert image to blur
    image_blur = blur(image).reshape(image.shape)
    # save image
    imsave(output_name, image_blur)

if __name__ == '__main__':
    serial()