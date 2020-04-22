import numpy as np
import matplotlib as plt


# returns image training and testing parameters 
def get_img_parameters(image, x_size):
    total_grey = 0
    top_grey = 0
    bot_grey = 0
    left_grey = 0
    right_grey = 0
    img_size = len(image)
    mid_picture = img_size//2
    collumn_index = 0
    row_index = 0
    # Loop through every pixel and find total grey value for 
    # the top, bottom, left, right and the whole picture 
    for x in range(image):
        collumn_index = x - row_index*x_size
        if collumn_index == x_size -1:
            row_index += 1
            collumn_index = x - row_index*x_size

        if image[x] > 10:
            total_grey += 255

            if x < mid_picture:
                top_grey += 255
            else:
                bot_grey += 255
            
            if collumn_index < x_size//2:
                left_grey += 255
            else:
                right_grey += 255
    
    top_bot_ratio = top_grey/bot_grey
    left_right_ratio = left_grey/right_grey
    mean_greyscale = total_grey/img_size

    return mean_greyscale, top_bot_ratio, left_right_ratio,  


def plot_number(image):
    image = np.reshape(28,28)
    plt.imshow(image)
          
def get_mean_position_for_number_list(number_list):
