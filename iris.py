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
    samples, labels = load_dataset()

    # Map indices of features to human friendly names:
    features = {0: 'sepal length',
                1: 'sepal width',
                2: 'petal length',
                3: 'petal width'}
    plot_histograms(samples, labels, features)

main()