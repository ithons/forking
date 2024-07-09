import numpy as np
import matplotlib.pyplot as plt

class table():
    def __init__(self):
        self.dims = (5, 3)
        self.square = {'width': (.5, 1.5), 'length': (1, 2), 'counter': 0}
        self.circle = {'center': (3, 1.5), 'radius': 1, 'counter': 0}

    def is_inside(self, x, y):
        if self.square['width'][0] <= x <= self.square['width'][1] and self.square['length'][0] <= y <= self.square['length'][1]:
            self.square['counter'] += 1
            return True, 'square'
        if (x - self.circle['center'][0])**2 + (y - self.circle['center'][1])**2 <= self.circle['radius']**2:
            self.circle['counter'] += 1
            return True, 'circle'
        return False
    


if __name__ == '__main__':
    t = table()
    ratios = []

    fig, ax = plt.subplots(5, 2, figsize=(18, 11))
    for i in range(1, 7):
        for j in range(10**i):
            ratio = t.circle['counter'] / t.square['counter'] if t.square['counter'] else 0
            ratios.append(ratio)
            x, y = np.random.uniform(0, t.dims[0]), np.random.uniform(0, t.dims[1])
            t.is_inside(x, y)
        ax[(i-1)//2, (i-1)%2].plot(ratios)
        ax[(i-1)//2, (i-1)%2].set_title(f'Iteration {j+1} with ratio {ratio:.2f}')
        if j > 1000:
            ax[(i-1)//2, (i-1)%2].set_xlim(j-1000, j)
            ax[(i-1)//2, (i-1)%2].set_ylim(2.5, 4)
    plt.tight_layout()
    plt.show()
