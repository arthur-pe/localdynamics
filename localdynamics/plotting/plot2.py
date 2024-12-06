import numpy as np
import matplotlib

def plot_sv_barcode(ax, nb_neighbors, sv, dim=None, cmap1=matplotlib.colormaps['Blues_r'],
                    cmap2=matplotlib.colormaps['Greens']):

    #print(np.nanmax(sv, axis=-1, keepdims=True))
    sv = sv / np.nanmax(sv, axis=-1, keepdims=True)
    # sv = sv/jnp.nanmax(sv, axis=(0,))
    # sv = jnp.nanquantile(sv, 0.99, axis=-2)
    sv = np.nanmax(sv, axis=-2)
    sv = sv.T

    dim = sv.shape[-2] if dim is None else dim

    for si, s in enumerate(sv):
        color = cmap1((si + 1) / dim) if si < dim else cmap2((si + 1) / (len(sv) - dim))

        ax.plot(nb_neighbors, s, '-o', color=color if si != 0 else (0.2, 0.2, 0.2), zorder=4)

    ax.set_xlabel('# Neighbors'), ax.set_ylabel('Local Dyn. SV')


def plot_srank(ax, nb_neighbors, sv, dim=None):
    sv = sv / np.nanmax(sv, axis=-1, keepdims=True)
    # sv = sv/jnp.nanmax(sv, axis=(0,))
    sv = np.nanquantile(sv, 1.0, axis=-2)
    sv_ = np.nansum(sv ** 4, axis=-1)

    # sv  = jnp.exp(-jnp.nansum(jnp.log(sv/sv_[..., jnp.newaxis])*sv/sv_[..., jnp.newaxis], axis=-1))
    sv = sv_
    # sv = sv[1:]
    # sv = jnp.nanargmax(jnp.abs(jnp.arange(sv.shape[-1])[jnp.newaxis]/(sv.shape[-1] - 1) + (sv - jnp.nanmin(sv, axis=-1, keepdims=True))/(jnp.nanmax(sv, axis=-1, keepdims=True) - jnp.nanmin(sv, axis=-1, keepdims=True)) - 1)-1, axis=-1)/jnp.sqrt(2)

    # sv = sv[1:] - sv[:-1]

    ax.plot(nb_neighbors, sv, '-o', color='red')

    if dim is not None:
        ax.axhline(dim, linestyle='--', label='True dim')
        ax.legend()

    ax.set_xlabel('# Neighbors'), ax.set_ylabel('sRank')


def plot_latent(ax, latent):

    if latent.shape[-1]>3:
        _, _, V = np.linalg.svd(latent, full_matrices=False)
        latent = latent @ V[:3].T
        ax.set_xlabel('PC1'), ax.set_ylabel('PC2'), ax.set_zlabel('PC3')
    else:
        ax.set_xlabel('Var. 1'), ax.set_ylabel('Var. 2'), ax.set_zlabel('Var. 3')

    ax.plot(*latent[:, :3].T, color=(0.95, 0.8, 0.2))