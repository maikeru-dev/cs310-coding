import numpy as np
import matplotlib.pyplot as plt

data_path = "./"

trainIn = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")

TARGETDIGIT = 7
trainIn = trainIn[trainIn[:, 0] == TARGETDIGIT]
trainLabel = [int(d[0] == TARGETDIGIT) for d in trainIn]


fig = plt.figure(
    figsize=(
        4,
        4,
    )
)
# data = p.weights[1:].reshape(28, 28)
vis = trainIn[0, 1:].reshape(28, 28)
plt.imshow(vis)
plt.show()
