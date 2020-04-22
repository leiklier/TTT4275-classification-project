import numpy as np
import pickle as pkl

# Training datasets
data_train = np.genfromtxt('./datasets/mnist_train.csv', delimiter=",")

images_train = [[], [], [], [], [], [], [], [], [], []]

i = 0
for image in data_train:
    if i >= 1000:
        break
     # Label is stored as first element
    label = np.int8(image[0])
    # Remove label from image
    image = np.array(image[1:], dtype=np.int8)
    images_train[label].append(image)
    i += 1

images_train = np.array(images_train)

with open('./datasets/mnist_train.pkl', 'wb') as file_train:
    pkl.dump(images_train, file_train)

print('Done with processing training datasets')
