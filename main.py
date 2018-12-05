from hive import Hive
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import time


def render_path(terrain, path):
    x, y = zip(*path)
    plt.plot(x, y, 'r+')
    plt.imshow(terrain)
    plt.show()


if __name__ == '__main__':
    # формируем случайный ландшафт,'низины' выравниваем, оставляем только 'горы'
    terrain = np.random.uniform(0, 1.0, (40, 40))
    terrain = np.where(terrain > 0.7, terrain, 0)
    terrain = scipy.signal.medfilt2d(terrain)

    hive = Hive(n_scouts=30,
                n_points=20,
                n_elite=10,
                n_bees_elite=50,
                n_bees_others=3,
                search_neighborhood=10,
                terrain=terrain,
                pos=(0, 0))

    start = time.time()
    solution = hive.find(target_pos=(39, 39), n_iters=100)

    end = time.time()
    print(end - start)

    render_path(terrain, solution.path)

