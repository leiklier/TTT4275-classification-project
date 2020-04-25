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

# Load the first num_samples images from train dataset with  corresponding labels:
def load_train_dataset(num_samples=10000):
    # Maximum 60 000 samples
    MAX_TRAIN_SAMPLES = 60000
    num_samples = num_samples % MAX_TRAIN_SAMPLES

    # Read from file
    raw_train = read_idx('./mnist_datasets/train-images.idx3-ubyte')
    train_labels = read_idx('./mnist_datasets/train-labels.idx1-ubyte')

    # Flatten images
    train_data = np.reshape(raw_train, (60000, 28*28))

    # Truncate  size to num_samples
    train_data = train_data[:num_samples]
    train_labels = train_labels[:num_samples]

    return (train_data, train_labels)

# Load the first num_samples images from test dataset with  corresponding labels:
def load_test_dataset(num_samples):
    # Maximum 10000 samples
    MAX_TEST_SAMPLES = 10000
    num_samples = num_samples % MAX_TEST_SAMPLES

    # Read from file
    raw_test = read_idx('./mnist_datasets/t10k-images.idx3-ubyte')
    test_labels = read_idx('./mnist_datasets/t10k-labels.idx1-ubyte')

    # Flatten images
    test_data = np.reshape(raw_test, (MAX_TEST_SAMPLES, 28*28))

    # Truncate size to num_samples
    test_data = test_data[:num_samples]
    test_labels = test_labels[:num_samples]

    return (test_data, test_labels)

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

def plot_misclassified_images(images, labels_true, labels_predicted, num_images=12):
    # Indices of some misclassified images
    image_indices = np.where(labels_true != labels_predicted)[0]
    image_indices = image_indices[:num_images]

    # We want a square layout of the images
    num_cols = np.uint16(np.floor(np.sqrt(num_images)))
    num_rows = np.uint16(num_images / num_cols)

    fig = plt.figure(figsize=(5, 5))
    fig.suptitle("Some misclassified images")

    for row in range(num_rows):
        for col in range(num_cols):
            i = (row * num_cols) + col
            image_index = image_indices[i]

            ax = fig.add_subplot(num_rows, num_cols, i+1)
            ax.axis("off")
            ax.set_title(f'T:{labels_true[image_index]}; P:{labels_predicted[image_index]}')

            image = np.reshape(images[image_index, :], (28,28))

            img_plot = ax.imshow(image)
            img_plot.set_interpolation("nearest")

def plot_correctly_classified_images(images, labels_true, labels_predicted):
    idx_9 = np.where((labels_true == 9) & (labels_predicted == 9))[0]
    idx_2 = np.where((labels_true == 2) & (labels_predicted == 2))[0]
    idx = np.reshape(np.array([idx_9, idx_2]), len(idx_9) + len(idx_2))

    fig = plt.figure(figsize=(5, 30))
    for i in range(len(idx)):
        ax = fig.add_subplot(len(idx), 1, i+1)
        image = np.reshape(images[idx[i], :], (28,28))
        img_plot = ax.imshow(image)
        img_plot.set_interpolation("nearest")

def plot_confusion_matrix(confusion_matrix, name):
    df_cm = pd.DataFrame(confusion_matrix, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
    fig = plt.figure(figsize = (10,7))

    fig.suptitle(name, fontsize=16)

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

        if predicted_number != test_labels[i]:
            num_errors += 1

    error_rate = num_errors/len(test_data)*100
    print("Error rate for NN using whole dataset:", error_rate, "%")

    confusion_matrix = metrics.confusion_matrix(test_labels, test_predicted)
    plot_confusion_matrix(confusion_matrix, name="CM for NN using whole dataset")

    #plot_correctly_classified_images(train_data, test_labels, test_predicted)
    plot_misclassified_images(train_data, test_labels, test_predicted)




def run_k_nearest_neighbour_classifier(train_data, train_labels, test_data, test_labels):
    num_errors = 0
    test_predicted = []
    NUM_NEIGHBOURS = 4

    for i in range(len(test_data)):
        nearest_neighbours = k_nearest_neighbours(train_data, train_labels, test_data[i], NUM_NEIGHBOURS)
        predicted_number = classify_img(nearest_neighbours)
        test_predicted.append(predicted_number)

        if predicted_number != test_labels[i]:
            num_errors += 1

    error_rate = num_errors/len(test_data)*100
    print("Error rate for KNN with k =",NUM_NEIGHBOURS, "using whole dataset:", error_rate,"%")

    confusion_matrix = metrics.confusion_matrix(test_labels, test_predicted)
    plot_confusion_matrix(confusion_matrix, name="CM FOR KNN using whole dataset")



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
    print("Error rate for NN using cluster:", error_rate, "%")

    confusion_matrix = metrics.confusion_matrix(test_labels, test_predicted)
    plot_confusion_matrix(confusion_matrix, name="CM for NN using cluster")

def main():

    TRAIN_SIZE = 1000
    TEST_SIZE = 100

    train_data, train_labels = load_train_dataset(num_samples=TRAIN_SIZE)
    test_data, test_labels = load_test_dataset(num_samples=TEST_SIZE)

    run_nearest_neighbour_classifier(train_data, train_labels, test_data, test_labels)
    run_k_nearest_neighbour_classifier(train_data, train_labels, test_data, test_labels)
    run_cluster_classifier(train_data, train_labels, test_data, test_labels)

    plt.show()

main()