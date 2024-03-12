import jax
from jaxtyping import Array, Float
import jax.numpy as jnp

def inverse_gamma_sample(key, 
                         alpha,
                         beta,
                         shape):
    """
    Generate samples from an inverse-gamma distribution using JAX.
    
    Parameters:
    - key: a PRNGKey used as the random key.
    - alpha: the shape parameter of the inverse-gamma distribution.
    - beta: the rate parameter of the inverse-gamma distribution.
    - shape: the shape of the output sample array.

    Returns:
    - Samples from the inverse-gamma distribution.
    """
    gamma_samples = jax.random.gamma(key, alpha, shape=shape) / beta
    return 1.0 / gamma_samples

def gibbs_sample_horseshoe(key, 
                           taus_sq: Float,
                           lambdas_sq: Float[Array, "p"],
                           betas: Float[Array, "p"],
                           p: int):

    """This does the horseshoe sampling using an auxilliary variable, giving conjugate inverse gamma
    See: A simple estimator for horseshoe
    """
    keys = jax.random.split(key, 5)
    # Sample from the auxilliary variables
    xi = inverse_gamma_sample(keys[0], 
                              1.0, 1.0 + 1/taus_sq, 
                              shape=(1,))

    vs = inverse_gamma_sample(keys[1],
                              jnp.ones(p),
                              1.0 + 1/lambdas_sq,
                              shape=(p,))

    # Sample from the hyperpriors
    taus_sq = inverse_gamma_sample(keys[2],
                                   (p+1)/2 ,
                                   1/xi + 0.5 * jnp.sum(betas ** 2 / lambdas_sq),
                                   shape=(1,))
    
    lambdas_sq = inverse_gamma_sample(keys[3],
                                      jnp.ones(p),
                                      1/vs + 0.5*(betas ** 2 / taus_sq),
                                      shape=(p,))

    return xi, vs, taus_sq, lambdas_sq