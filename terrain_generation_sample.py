import numpy as np
import matplotlib.pyplot as plt
import scipy.signal


if __name__ == '__main__':
    terrain = np.random.randn(40, 40)
    plt.subplot(1, 3, 1)
    plt.imshow(terrain)

    terrain = scipy.signal.medfilt(terrain)
    plt.subplot(1, 3, 2)
    plt.imshow(terrain)

    terrain = scipy.signal.medfilt2d(terrain)
    arr = [1, 2, 4]
    plt.plot(arr, 'r+')
    plt.subplot(1, 3, 3)
    plt.imshow(terrain)
    plt.show()