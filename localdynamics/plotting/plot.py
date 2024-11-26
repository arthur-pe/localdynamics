import numpy as np

from matplotlib import pyplot as plt
import matplotlib

stable_rank = lambda s: (s ** 2).sum(axis=-1) / s[..., 0] ** 2


def get_axes_list(number_axes, axes_3d, figsize=4):
  '''
  args:
    numbe_axes: number of ax in the list
    axes_3d: indices of 3d axes
  return:
    fig: figure
    axs: list of ax
  '''

  fig = plt.figure(figsize=(number_axes*figsize, figsize), constrained_layout=True)
  axs = [fig.add_subplot(1, number_axes, i+1, projection='3d' if i in axes_3d else None) for i in range(number_axes)]

  return fig, axs


def get_axes_grid(cols, rows, dpi=100):

    fig = plt.figure(figsize=(cols*4,rows*4), constrained_layout=True, dpi=dpi)
    gs = fig.add_gridspec(ncols=cols,nrows=rows)

    axs = np.array([[fig.add_subplot(gs[i,j]) for i in range(rows)] for j in range(cols)])

    return fig, np.array(axs)


def plot_spectra(axs, Ss, log=False, normalize=True, quantile=0.1, plotall=False, color='red', label=None):

    if normalize: Ss = [s/np.max(np.median(s, axis=0)) for s in Ss]

    for ni, s in enumerate(Ss):
        ids = np.arange(s.shape[-1]) + 1
        axs[ni].plot(ids, np.median(s, axis=0), 'o-', color=color, linewidth=2.0, zorder=3)
        if plotall:
            for s_ in s:
                for s in s_: axs[ni].plot(ids, s, alpha=0.1, color=color, linewidth=0.5, zorder=1)
        axs[ni].fill_between(ids, np.quantile(s, quantile, axis=0),
                                np.quantile(s, 1 - quantile, axis=0),
                                color=color, alpha=0.1, zorder=2,
                                label=f'Quantile: {quantile} - {1 - quantile}\n{label if label is not None else ""}')

        axs[ni].set_xlabel('s.v. index')
        if log: axs[ni].set_yscale('log')

        if ni == 0:
            axs[ni].set_ylabel('singular value')
            axs[ni].legend()


def plot_effective_rank(axs, Ss, erank_function=stable_rank, bins=50, color='red', alpha=0.3, label=None):


        for ni, s in enumerate(Ss):

            erank = erank_function(s)

            axs[ni].hist(erank.reshape(-1), bins=bins, color=color, alpha=alpha, label=label, density=True)
            axs[ni].axvline(np.mean(erank), linewidth=2.0, linestyle='--', color=color, label=label)
            axs[ni].set_xlabel('Effective rank')
            if ni == 0:
                axs[ni].set_ylabel('Fraction of points')
                if label is not None: axs[ni].legend()
