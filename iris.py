import numpy as np
import matplotlib.pyplot as plt

def load_dataset():
    sample_label_pairs = [] # [ (data, label) ]

    with open(f'./iris_dataset.csv', 'rb') as csv_file:
        for row in csv_file:
            cells = [ cell.strip() for cell in row.decode().split(',') ]

            label = cells[ len(cells) - 1 ]
            sample = np.array(cells[:len(cells) - 1], dtype=np.float32)

            sample_label_pairs.append( (sample, label) )

    sample_label_pairs = np.array(sample_label_pairs)
    np.random.shuffle(sample_label_pairs)

    samples, labels = np.transpose(sample_label_pairs)
    samples = np.array(samples)
    labels = np.array(labels)

    return samples, labels

def split_dataset(samples, labels, split_index):
    first_set = samples[:split_index], labels[:split_index]
    last_set = samples[split_index:], labels[split_index:]

    return first_set, last_set


######################################
#      -- LINEAR CLASSIFIER --       #
######################################

# Implements eq. (3.19) using
# predictions[k] = g_k
# targets[k] = t_k
def MSE(predictions, targets):
    error = predictions-targets
    error_T = np.transpose(error)
    return np.sum(np.matmul(error_T,error)) / predictions.shape[0]

# Implements eq. (3.20) using
# samples = x
def get_predicted_labels(samples, W):
    exponentials = np.array([ np.exp(-(np.matmul(W, sample))) for sample in samples ])
    denominators = exponentials + 1
    predictions = 1 / denominators

    return predictions

# Implements eq. (3.22) and eq. (3.23) using
# predicted_labels[k] = g_k
# labels[k] = t_k
# samples = x
# previous_W = W(m - 1)
# ASSUMES: len(predicted_labels) = len(labels) = len(samples)
def iterate_weight_matrix(predicted_labels, labels, samples, previous_W, alpha=0.01):
    num_features = len(samples[0]) - 1 # Subtract 1 because of the 1-fill
    num_classes = 3
    grad_g_MSE = predicted_labels - labels # dim (30,3)
    grad_z_g = predicted_labels * (1 - predicted_labels) # dim (30,3)
    # print(predicted_labels)
    grad_W_z = np.array([ np.reshape(sample, (1, num_features+1)) for sample in samples ])

    grad_W_MSE = np.sum( np.matmul(np.reshape(grad_g_MSE[k] * grad_z_g[k], (num_classes, 1)), grad_W_z[k]) for k in range(len(grad_g_MSE)) )

    next_W = previous_W - alpha * grad_W_MSE

    return next_W

def train_linear_classifier(samples, labels, features, num_iterations=100, alpha=0.01):
    classes = np.unique(labels)
    num_classes = 3
    num_features = len(features)

    # Initialize weight matrix
    print("Num classes:", num_classes)
    print("Num features:", num_features)
    W = np.zeros((num_classes, num_features+1))

    for curr_iteration in range(num_iterations):
        predicted_labels = get_predicted_labels(samples, W)
        W = iterate_weight_matrix(predicted_labels, labels, samples, W, alpha)

    return W

def get_error_rate(predicted_labels, labels):
    num_samples = len(labels)

    num_errors = 0
    for i in range(len(labels)):
        if labels[i] != predicted_labels[i]:
            num_errors += 1

    return num_errors / num_samples

def get_confusion_matrix(predicted_labels, labels):
    classes = np.unique(labels)

    confusion_matrix = []
    for predicted_class in classes:
        row = []

        for true_class in classes:
            # All occurences of current true_class in labels_true:
            true_indices = np.where(labels == np.uint8(true_class))[0]

            # All occurences of current predicted_class in labels_predicted:
            predicted_indices = np.where(predicted_labels == np.uint8(predicted_class))[0]

            # We want to find the number of elements where these two matches:
            num_occurences = len( np.intersect1d(true_indices, predicted_indices) )
            row.append(num_occurences)

        confusion_matrix.append(row)

    return np.array(confusion_matrix)

def label_string_to_vector(string_label, classes):
    index = np.where(classes == string_label)[0]
    vector_label = np.array([ i == index for i in range(len(classes)) ], dtype=np.uint8)
    vector_label = np.reshape(vector_label, len(vector_label))

    return vector_label

def label_vector_to_string(vector_label, classes):
    index = np.argmax(vector_label)
    vector_string = classes[index]

    return vector_string

#######################################

def plot_histograms(samples, labels, features, step_length=0.1):
    classes = np.unique(labels)

    num_subplots = len(features)
    num_cols = np.uint8(np.floor(np.sqrt(num_subplots)))
    num_rows = np.uint8(np.ceil(num_subplots / num_cols))

    fig = plt.figure(figsize=(10, 10))

    for feature_index in range(len(features)):
        feature = features[feature_index]

        ax = fig.add_subplot(num_cols, num_rows, feature_index+1)
        ax.set(xlabel='Measurement [cm]', ylabel='Number of samples')

        for curr_class in classes:
            sample_indices = np.where(labels == curr_class)[0]

            samples_matching_class = [ samples[i] for i in sample_indices ]
            measurements_by_features = np.transpose(samples_matching_class)
            measurements_matching_feature = measurements_by_features[feature_index]

            ax.hist(measurements_matching_feature, alpha=0.5, stacked=True, label=curr_class)

        ax.set_title(feature)
        ax.legend(prop={'size': 7})

    plt.show()


def main():
    all_samples, all_labels = load_dataset()

    # Map indices of features to human friendly names:
    features = {0: 'sepal length',
                1: 'sepal width',
                2: 'petal length',
                3: 'petal width'}
    # plot_histograms(all_samples, all_labels, features)

    train_dataset, test_dataset = split_dataset(all_samples, all_labels, 30)
    train_samples, train_labels = train_dataset
    test_samples, test_labels = test_dataset

    # We need to add an awkward 1 to x_k as described on page 15:
    train_samples = np.array([ np.append(sample, [1]) for sample in train_samples ])
    test_samples = np.array([ np.append(sample, [1]) for sample in test_samples ])

    classes = np.unique(train_labels) # Navn på de ulike typene blomster

    train_label_vectors = np.array([ label_string_to_vector(label, classes) for label in train_labels])
    test_label_vectors = np.array([ label_string_to_vector(label, classes) for label in test_labels])

    W = train_linear_classifier(train_samples, train_label_vectors, features, num_iterations=1000)

    predicted_test_label_vectors = get_predicted_labels(test_samples, W)

    print("Predicted:", predicted_test_label_vectors[5])
    print("Actual:", test_label_vectors[5])
    print(np.argmax(predicted_test_label_vectors[5]))

    predicted_test_label_strings = np.array([ label_vector_to_string(label, classes) for label in predicted_test_label_vectors])

    error_rate = get_error_rate(predicted_test_label_strings, test_labels)

    print("Error rate:", error_rate)
    print("Predicted:", predicted_test_label_strings[20:30])
    print("Actual:", test_labels[20:30])

main()