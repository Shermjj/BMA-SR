import equinox as eqx
import jax
import jax.numpy as jnp
import diffrax
import numpy as np
import optax

def lipswish(x):
    return 0.909 * jax.nn.silu(x)

def dx_dt_lorenz(x, f=10):
    """"
    Note: x is just a single vector of shape (shape_size/param_size)
    as opposed to previous implementations which was a tensor of shape (batch_size, shape_size/param_size)

    This computes the time derivative for the non-linear deterministic Lorenz 96 Model of arbitrary dimension n.
    dx/dt = f(x) 
    """
    # shift minus and plus indices
    x_m_2 = jnp.concatenate([x[-2:], x[:-2]])
    #print(x_m_2)
    x_m_1 = jnp.concatenate([x[-1:], x[:-1]])
    #print(x_m_1)
    x_p_1 = jnp.concatenate((x[1:], x[0:1]))
    #print(x_p_1)

    dxdt = (x_p_1-x_m_2)*x_m_1 - x + f

    return dxdt


class Lorenz_VectorField(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, *, key, **kwargs):
        super().__init__(**kwargs)
        scale_key, mlp_key = jax.random.split(key)
        self.mlp = eqx.nn.MLP(
            in_size=8,
            out_size=8,
            width_size=6,
            depth=1,
            activation=lipswish,
            #final_activation=jax.nn.tanh,
            key=mlp_key
        )


    def __call__(self, t, y, args):
        return dx_dt_lorenz(y) - self.mlp(y)
        #return self.mlp(y)

class NeuralSDE(eqx.Module):
    vf: Lorenz_VectorField  # drift
    sigma: jax.Array  # diffusion
    readout: eqx.nn.Linear
    tol: float
    #y0: jax.Array

    def __init__(
        self,
        *,
        key,
        tol=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        #self.y0 = y0
        initial_key, vf_key, cvf_key, readout_key = jax.random.split(key, 4)
        self.vf = Lorenz_VectorField(key=vf_key) # Drift term
        # By default this is in=8, out=8, width=6, depth=1

        self.sigma = jnp.array([2.0]) # Start from 2.0

        self.readout = eqx.nn.Linear(8, 8, key=readout_key)
        # Final readout layers in=8, out=8

        self.tol = tol

    def __call__(self, ts=jnp.linspace(0, 1.5, 21), *, key):
        t0 = ts[0]
        t1 = ts[-1]

        # Very large dt0 for computational speed
        dt0 = self.tol

        control = diffrax.VirtualBrownianTree(
            t0=t0, t1=t1, tol=1e-3, shape=(8,), key=key
        )
        vf = diffrax.ODETerm(self.vf)  # Drift term
        cvf = diffrax.ControlTerm(lambda t, y, args: self.sigma * y, control)  # Diffusion term
        terms = diffrax.MultiTerm(vf, cvf)
        # ReversibleHeun is a cheap choice of SDE solver. We could also use Euler etc.
        solver = diffrax.Euler()
        saveat = diffrax.SaveAt(ts=ts)

        # hardcoding initial values for now
        # TODO: Figure out how to make this a class attribute without breaking diffrax and taking grads
        # Use https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.stop_gradient.html
        y0 = jnp.array([ 8.219852 ,  1.5055572, -0.471462 ,  2.0423932,  3.058058 , 1.4114221,  6.6189885,  6.5770426])
        sol = diffrax.diffeqsolve(terms, solver, t0, t1, self.tol, y0, saveat=saveat)
        #print(sol.ys)
        #print(f"sol.ys.shape: {sol.ys.shape}")
        ys = jax.vmap(self.readout)(sol.ys[1:])
        #ys = sol.ys
        #print(f"ys.shape: {ys.shape}")  
        ys = (ys - y0.mean()) / y0.std()
        return ys