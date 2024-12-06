import numpy as np
import pickle
from  math import ceil

def import_data(max_time_dim=2000):

    with open('./data/fish/data_150fish.pkl', 'rb') as f:
        data = pickle.load(f)

    data = data[::ceil(data.shape[0]/max_time_dim)]

    data = data.reshape(data.shape[0], -1)

    return [{'name':'Fish', 'xs':data, 'ts':np.linspace(0, 1, data.shape[0])}]


if __name__ == '__main__':

    import_data()