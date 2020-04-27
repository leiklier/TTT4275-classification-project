
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, num=1000)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def step(x):
    y = []
    for i in range(len(x)):
        if x[i] > 0:
            y.append(1)
        else:
            y.append(0)
    return np.array(y)

class_1_x = np.array([-7.5, -5, -1, 1.2])
class_1_y = [ 0 for i in range(len(class_1_x)) ]

class_2_x = np.array([0, 5, 7])
class_2_y = [ 1 for i in range(len(class_2_x))]

plt.scatter(class_1_x, class_1_y, label="$c_1$ samples", c='r')
plt.scatter(class_2_x, class_2_y, label="$c_2$ samples",  c='b')

plt.plot(x, sigmoid(x), label="$s(x) = \\frac{1}{1 + e^{-x}}$")
plt.plot(x, step(x), label="$u(x)$")

plt.legend()
plt.xlabel("Measurement value")
plt.ylabel("Class likelihood score")

plt.savefig('./images/squashing_functions.png')
# plt.show()
