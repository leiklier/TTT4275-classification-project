import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import struct
from scipy.spatial import distance
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

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



def classify_img(nearest):
    nearest_number_counts = [0]*10
    highest = 0
    number = -1
    for i in range(len(nearest)):
        index = nearest[i][1]
        nearest_number_counts[index] += 1
    for i in range(len(nearest_number_counts)):
        if nearest_number_counts[i] > highest:
            number = i

    return number



def nn_two(n, data, data_labels, image):
    dist_and_indx_list = []
    for i in range (len(data)):
        dist = distance.euclidean(data[i], image)
        dist_and_indx_list.append((dist, data_labels[i]))
    dist_and_indx_list = sorted(dist_and_indx_list, key=lambda dist: dist[0])
    return dist_and_indx_list[:n]
    
def main():

    raw_train = read_idx('./datasets/train-images.idx3-ubyte')
    train_data = np.reshape(raw_train, (60000, 28*28)) # Flatten images
    train_labels = read_idx('./datasets/train-labels.idx1-ubyte')

    raw_test = read_idx('./datasets/t10k-images.idx3-ubyte')
    test_data = np.reshape(raw_test, (10000, 28*28)) # Flatten images
    test_labels = read_idx('./datasets/t10k-labels.idx1-ubyte')

    train_data = train_data[:60000]
    train_labels = train_labels[:60000]

    test_data = test_data[:100]
    test_labels = test_labels[:100]

    # Kjørr
    test_predicted = np.array([0]*len(test_labels), dtype=np.uint8) 
    erorr = 0
    for i in range(len(test_data)):
        n_nearest = nn_two(4, train_data, train_labels, test_data[i])
        predicted_label = classify_img(n_nearest)
        
        test_predicted[i] = predicted_label
        if predicted_label != test_labels[i]:
            erorr +=1 
    errorrate = erorr/len(test_data)*100
    print("Errorate =",errorrate, "%")
    confusion_matrix = metrics.confusion_matrix(test_labels, test_predicted)
    print(confusion_matrix)

main()