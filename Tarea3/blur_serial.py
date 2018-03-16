import numpy as np
from scipy.misc import imread, imsave

def product(vector):
    p = 1
    for vi in vector:
        p *= vi
    return p

def blur(image):
    image_normal = image.reshape(product(image.shape))
    image_width, image_height, _ = image.shape
    image_blur = np.zeros_like(image).reshape(product(image.shape))
    BLUR_SIZE = 5

    for col_i in range( image_width ):
        for row_i in range( image_height ):
            indexI3 = (row_i * image_width + col_i)*3
            pixValR, pixValG, pixValB, pixels = (0,0,0,0)
            
            for blurRow in range(-BLUR_SIZE, BLUR_SIZE+1):
                for blurCol in range(-BLUR_SIZE, BLUR_SIZE+1):
                    curRow, curCol = row_i + blurRow, col_i + blurCol
                    indexB3 = (curRow * image_width + curCol)*3
                    # Verify we have a valid image pixel
                    if (image_height > curRow > -1) and (image_width > curCol > -1):
                        pixValR += image_normal[indexB3]
                        pixValG += image_normal[indexB3+1]
                        pixValB += image_normal[indexB3+2]
                        pixels += 1 # Keep track of number of pixels in the accumulated total
            # Write our new pixel value out
            image_blur[indexI3] = pixValR/pixels
            image_blur[indexI3+1] = pixValG/pixels
            image_blur[indexI3+2] = pixValB/pixels

    return image_blur
    

def serial(input_name='test_3.jpeg', output_name='test_blur_3.jpeg'):
    image = imread(input_name)
    # convert image to blur
    image_blur = blur(image).reshape(image.shape)
    # save image
    imsave(output_name, image_blur)

if __name__ == '__main__':
    serial()