import numpy as np
import scipy
from math import ceil

def import_data_one_animal(animal, max_time_dim=2000):

    data = scipy.io.loadmat(f'./data/saxena/{animal}_cn_new.mat', simplify_cells=True)

    data = data[list(data.keys())[-1]]

    #for k in ['velocity', 'accel', 'horzPos', 'horzVel', 'vertPos', 'vertVel']: data[k] = data[k].T


    xa = data['xA'][:data['xA'].shape[0]//2] ; xa = xa[::ceil(xa.shape[0]/max_time_dim)]
    za = data['zA'][:data['zA'].shape[0]//2] ; za = za[::ceil(za.shape[0]/max_time_dim)]

    return [{'name':f'Motor cortex (Monkey {animal})', 'xs':xa, 'ts':np.linspace(0, 16, xa.shape[0])},
            {'name':f'Muscle (Monkey {animal})', 'xs':za, 'ts':np.linspace(0, 16, za.shape[0])}]


def import_data(max_time_dim=2000):

    animals = ['Cousteau', 'Drake']

    return import_data_one_animal(animals[0], max_time_dim) + import_data_one_animal(animals[1], max_time_dim)


if __name__ == '__main__':

    print(ceil(0.1))