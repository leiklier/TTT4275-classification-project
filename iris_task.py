#%%

import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

def load_data():
    class_1 = []
    class_2 = []
    class_3 = []
    f = open("iris_datasets/class_1")
    for line in f:
        temp = line[:15]
        temp = temp.split(",")
        class_1.append(temp)
    f.close()
    f = open ("iris_datasets/class_2")
    for line in f:
        temp = line[:15]
        temp = temp.split(",")
        class_2.append(temp)
    f.close()
    f = open ("iris_datasets/class_3")
    for line in f:
        temp = line[:15]
        temp = temp.split(",")
        class_3.append(temp)
    f.close()

    return class_1, class_2, class_3



class_1, class_2, class_3 = load_data()
data_sets = [class_1, class_2]
for dataset in data_sets:
    training_data = dataset[:30]
    test_data = dataset[30:]

    for t in test_data:
        petal_w = float(t[2])
        petal_l = float(t[3])

        
        plt.scatter(petal_w, petal_l)
    




# %%


# %%
