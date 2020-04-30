# Classification project in TTT4275

**Authors**: Bratvold, Torbjørn and Lima-Eriksen, Leik

**Subject**: [Estimation, Detection, and Classification (TTT4275)](https://www.ntnu.edu/studies/courses/TTT4275#tab=omEmnet)

**Date**: April 2020

---

The project is divided into two parts:

1. **IRIS dataset**: The performance of a linear classifier is thoroughly analyzed. Here, we take a deeper look on the impact of choosing the right samples for testing and training, and how creating a different partitioning may trick us into thinking that the classifier perfoms better when it in fact performs just as good.

    We then evaluate the performance after excluding some of the most overlapping features. Surprisingly, it turns out that the classifier has the same error rate but requires more computations. The reasons why are discussed in detail in the report.

2. **MNIST dataset**: Different variants of the K-Nearest Neighbours (KNN) classifier is thoroughly analyzed in terms of error rates, confusion matrices and computation times. We start with training a KNN classifier with K=1 on the dataset. Then we compare the performance against a KNN classifier with K=4. It turns out that the error rate is marginally better for K=4.

    In the last part we first apply a K-means clustering on the samples. Then we run a KNN classifier with K=1 on the test data. This results in 10 times less errors (5% error rate compared to 51%) and a lot less computations required.

## Prerequisities
Please install the prerequisities before running the scripts:

```
pip3 install -r requirements.txt
```