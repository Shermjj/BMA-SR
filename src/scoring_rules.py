import jax.numpy as jnp

def energy_score(s_observations, s_simulations):
    """
    We estimate this by building an empirical unbiased estimate of Eq. (2) in Ziel and Berk 2019

    Parameters
    ----------
    s_observations: numpy array
        The summary statistics of the observed data set. Shape is (n_obs, n_summary_stat).
    s_simulations: numpy array
        The summary statistics of the simulated data set. Shape is (n_sim, n_summary_stat).
    """
    n_obs = s_observations.shape[0]
    n_sim, p = s_simulations.shape
    diff_X_y = s_observations.reshape(n_obs, 1, -1) - s_simulations.reshape(1, n_sim, p)
    diff_X_y = jnp.einsum('ijk, ijk -> ij', diff_X_y, diff_X_y)

    diff_X_tildeX = s_simulations.reshape(1, n_sim, p) - s_simulations.reshape(n_sim, 1, p)

    # exclude diagonal elements which are zero:
    diff_X_tildeX = jnp.einsum('ijk, ijk -> ij', diff_X_tildeX, diff_X_tildeX)
    diff_X_tildeX = jnp.where(~jnp.eye(n_sim, dtype=bool), diff_X_tildeX, 0) #Remove diagonal elements

    beta_over_2 = 0.5
    diff_X_y **= beta_over_2
    diff_X_tildeX **= beta_over_2

    return 2 * jnp.sum(jnp.mean(diff_X_y, axis=1)) - n_obs * jnp.sum(diff_X_tildeX) / (
            n_sim * (n_sim - 1))
