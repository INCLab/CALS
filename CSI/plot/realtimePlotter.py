# Todo: real Time plotter 완성하기
import numpy as np
import matplotlib.pyplot as plt

x=0

for i in range(1000):
    x = x + 0.1
    y = np.sin(x)

    plt.scatter(x, y)
    plt.pause(0.001)

plt.show