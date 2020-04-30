import numpy as np
import matplotlib.pyplot as plt

def load_dataset():
    sample_label_pairs = [] # [ (data, label) ]

    with open(f'./iris_dataset.csv', 'rb') as csv_file:
        for row in csv_file:
            cells = [ cell.strip() for cell in row.decode().split(',') ]

            label = cells[ len(cells) - 1 ]
            sample = np.array(cells[:len(cells) - 1], dtype=np.float32)

            # Keep only petal length and width
            sample = sample[:2 ]

            sample_label_pairs.append( (sample, label) )

    sample_label_pairs = np.array(sample_label_pairs)
    np.random.shuffle(sample_label_pairs)

    samples, labels = np.transpose(sample_label_pairs)
    samples = np.array(samples)
    labels = np.array(labels)

    return samples, labels

samples, labels = load_dataset()
classes = np.unique(labels)

# Plot petal length and width
for curr_class in classes:
    indices = np.where(labels == curr_class)[0]
    petal_lengths = [ samples[i][0] for i in indices ]
    petal_widths = [ samples[i][1] for i in indices ]
    plt.scatter(petal_lengths, petal_widths, label=curr_class)

plt.xlabel("Sepal length [cm]")
plt.ylabel("Sepal width [cm]")

plt.legend()
plt.show()