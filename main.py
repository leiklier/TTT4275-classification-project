import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import cv2 
from scipy.spatial import distance

# Import data:
def load_train(): 
    data = []
    with open('./datasets/mnist_train.pkl', 'rb') as file_train:
        data = pkl.load(file_train)
    return np.array(data)

def load_test():
    data = []
    with open('./datasets/mnist_test.pkl', 'rb') as file_test:
        data = pkl.load(file_test)
    return np.array(data)

# returns image training and testing parameters 
def get_img_parameters(image, x_size, y_size):
    total_grey = 0
    top_grey = 0
    bot_grey = 0
    left_grey = 0
    right_grey = 0
    img_size = x_size*y_size
    mid_picture_height = y_size//2
    mid_picture_width = x_size//2
    # Loop through every pixel and find total grey value for 
    # the top, bottom, left, right and the whole picture 
    for y in range(y_size):
        for x in range(x_size):

            if image[y][x] > 10:
                total_grey += 255

                if y < mid_picture_height:
                    top_grey += 255
                else:
                    bot_grey += 255
                
                if x < mid_picture_width:
                    left_grey += 255
                else:
                    right_grey += 255
    
    # Returns mean picture greyscale and the ratio between grayscale in
    # top and bottom and the ratio between left and right
    top_bot_ratio = top_grey/bot_grey
    left_right_ratio = left_grey/right_grey
    mean_greyscale = total_grey/img_size

    return mean_greyscale, top_bot_ratio, left_right_ratio,  



          
def get_mean_position_for_number_list(number_list):
    x_mean = np.mean(number_list[0])
    y_mean = np.mean(number_list[1])
    z_mean = np.mean(number_list[2])
    number_list_mean_pos = [x_mean, y_mean, z_mean]
    return number_list_mean_pos

def get_distance_between_points(point_1, point_2):
    dist = distance.euclidean(point_1, point_2)
    return dist 

def train(train_data):
    parameter_list = []
    mean_pos_of_numbers = []
    for y in range(len(train_data)):
        for x in range(len(train_data[y])):
            img = train_data[y][x]
            par = get_img_parameters(img, 28, 28)
            parameter_list.append(par)
            if x == 100:
                break
        mean_pos = get_mean_position_for_number_list(parameter_list)
        mean_pos_of_numbers.append(mean_pos)
        parameter_list = []
    
    return mean_pos_of_numbers

def classify_number(mean_pos_of_numbers, img):
    nearest_dist = -1
    number = -1
    for i in range(len(mean_pos_of_numbers)):
        img_pos = get_img_parameters(img, 28, 28)
        dist = get_distance_between_points(mean_pos_of_numbers[i], img_pos)
        print(dist)
        if nearest_dist != -1:
            if dist < nearest_dist:
                nearest_dist = dist
                number = i
        else:
            nearest_dist = dist
            number = i
    return number

def test(mean_pos_of_numbers, test_data):
    nr_of_imgs = 100*10
    #for x in range(len(test_data)):
        #nr_of_imgs += len(test_data[x]) 

    errors = 0
    for i in range(len(test_data)):
        print("tested", i)
        for j in range(len(test_data[i])):
            img = test_data[i][j]
            number = classify_number(mean_pos_of_numbers, img)
            if number != i:
                errors += 1
            if j == 100:
                break
    errorate = errors/nr_of_imgs *100
    print(errorate,"%")


data = load_train()
#plt.imshow(data[3][2])
print("training")
pos = train(data)
#print(pos)
print("testing")
data = load_test()

n = classify_number(pos,data[0][10])
print(n)
#test(pos, data)
#plt.show()

