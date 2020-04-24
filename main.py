import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import struct
import seaborn as sn
import pandas as pd
from scipy.spatial import distance
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

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
        index = nearest[i]
        nearest_number_counts[index] += 1
    for i in range(len(nearest_number_counts)):
        if nearest_number_counts[i] > highest:
            number = i

    return number


def k_nearest_neighbours(data, data_labels, image, k_neighbours):
    dist_and_label_list = []
    for i in range (len(data)):
        dist = distance.euclidean(data[i], image)
        dist_and_label_list.append((dist, data_labels[i]))
    dist_and_label_list = sorted(dist_and_label_list, key=lambda dist: dist[0])
    
    labels = [ dist_and_label[1] for dist_and_label in dist_and_label_list[:k_neighbours] ]
    return labels
 
def get_clusters(dataset, labels, classes, n_clusters=64):
    label_and_clusters = [] # [ (label, clusters=[]) ] 
    for curr_class in classes:
        indices = labels == curr_class
        data_of_curr_class = dataset[indices]

        clusters = KMeans(n_clusters=n_clusters).fit(data_of_curr_class).cluster_centers_

        label_and_clusters.append( (curr_class, clusters) )

    return label_and_clusters
        
def get_cluster_dataset(dataset, labels):
    label_and_clusters = get_clusters(dataset, labels, classes=range(10), n_clusters=64)

    train_data = []
    train_labels = [ [i]*64 for i in range(10) ]
    train_labels = np.array(train_labels).reshape(64*10)
    for (label, clusters) in label_and_clusters:
        train_data.extend(clusters)
        
    return (train_data, train_labels)

def plot_misclassified_images(images, labels_predicted, labels_true):
    idx_9 = np.where((labels_true == 9) & (labels_predicted != 9))[0]
    idx_2 = np.where((labels_true == 2) & (labels_predicted != 2))[0]
    idx = np.reshape(np.array([idx_9, idx_2]), 6)

    fig = plt.figure(figsize=(5, 30))
    for i in range(len(idx)):
        ax = fig.add_subplot(len(idx), 1, i+1)
        image = np.reshape(images[idx[i], :], (28,28))
        img_plot = ax.imshow(image)
        img_plot.set_interpolation("nearest")

def plot_correctly_classified_images(images, labels_predicted, labels_true):
    idx_9 = np.where((labels_true == 9) & (labels_predicted == 9))[0]
    idx_2 = np.where((labels_true == 2) & (labels_predicted == 2))[0]
    idx = np.reshape(np.array([idx_9, idx_2]), len(idx_9) + len(idx_2))

    fig = plt.figure(figsize=(5, 30))
    for i in range(len(idx)):
        ax = fig.add_subplot(len(idx), 1, i+1)
        image = np.reshape(images[idx[i], :], (28,28))
        img_plot = ax.imshow(image)
        img_plot.set_interpolation("nearest")

def plot_confusion_matrix(confusion_matrix):
    df_cm = pd.DataFrame(confusion_matrix, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)

######################
# Runtime functions: #
######################
def run_nearest_neighbour_classifier(train_data, train_labels, test_data, test_labels):
    num_errors = 0
    test_predicted = []

    for i in range(len(test_data)):
        nearest_neighbours = k_nearest_neighbours(train_data, train_labels, test_data[i], 1)
        predicted_number = classify_img(nearest_neighbours)

        test_predicted.append(predicted_number)
        print(test_labels[i])

        if predicted_number != test_labels[i]:
            num_errors += 1

    error_rate = num_errors/len(test_data)*100
    print("Error rate:", error_rate, "%")

    confusion_matrix = metrics.confusion_matrix(test_labels, test_predicted)
    plot_confusion_matrix(confusion_matrix)

    #plot_correctly_classified_images(train_data, test_labels, test_predicted)
    #plot_misclassified_images(train_data, test_labels, test_predicted)
        
    
    

def run_k_nearest_neighbour_classifier(train_data, train_labels, test_data, test_labels):
    num_errors = 0
    test_predicted = []

    for i in range(len(test_data)):
        nearest_neighbours = k_nearest_neighbours(train_data, train_labels, test_data[i], 4)
        predicted_number = classify_img(nearest_neighbours)
        test_predicted.append(predicted_number)

        if predicted_number != test_labels[i]:
            num_errors += 1
            
    error_rate = num_errors/len(test_data)*100
    print("Error rate =",error_rate,"%")

    confusion_matrix = metrics.confusion_matrix(test_labels, test_predicted)
    plot_confusion_matrix(confusion_matrix)


    
def run_cluster_classifier(train_data, train_labels, test_data, test_labels):
    errors = 0
    train_cluster, train_labels = get_cluster_dataset(train_data, train_labels)
    test_predicted = []

    for i in range(len(test_data)):
        k_nearest = k_nearest_neighbours(train_cluster, train_labels, test_data[i], 1)
        predicted_number = classify_img(k_nearest)
        test_predicted.append(predicted_number)

        if predicted_number != test_labels[i]:
            errors +=1

    error_rate = errors/len(test_data)*100
    print("Error rate =",error_rate,"%")

    confusion_matrix = metrics.confusion_matrix(test_labels, test_predicted)
    plot_confusion_matrix(confusion_matrix)



        

def main():

    train_size = 60000
    test_size = 100

    raw_train = read_idx('./datasets/train-images.idx3-ubyte')
    train_data = np.reshape(raw_train, (60000, 28*28))
    train_data= train_data[:train_size]
    train_labels = read_idx('./datasets/train-labels.idx1-ubyte')
    train_labels = train_labels[:train_size]

    raw_test = read_idx('./datasets/t10k-images.idx3-ubyte')
    test_data = np.reshape(raw_test, (10000, 28*28))
    test_data = test_data[:test_size]
    test_labels = read_idx('./datasets/t10k-labels.idx1-ubyte')
    test_labels = test_labels[:test_size]

    

    run_nearest_neighbour_classifier(train_data, train_labels, test_data, test_labels)
    run_k_nearest_neighbour_classifier(train_data, train_labels, test_data, test_labels)
    run_cluster_classifier(train_data, train_labels, test_data, test_labels)

    plt.show()

main()