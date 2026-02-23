
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist #may need to see if change
import time
import jax.numpy as jnp
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_value,
)
from numpyro.handlers import condition

                      

def kernel(x1, x2, parameters, include_noise=True, sigma = 5e-2): 
    noise = sigma**2
    num_tasks = 3
    number_of_input_dimensions = 1

    if x1.ndim == 1:
        mesh = jnp.meshgrid(x1.flatten(), x2.flatten(), indexing='xy')
        xx, yy = jnp.expand_dims(mesh[0].T, axis=0), jnp.expand_dims(mesh[1].T, axis=0)

    elif x1.ndim == 2 and x1.shape[1] == number_of_input_dimensions:
        xx = jnp.zeros((number_of_input_dimensions, x1.shape[0], x2.shape[0]))
        yy = jnp.zeros((number_of_input_dimensions, x1.shape[0], x2.shape[0]))
        for i in range(number_of_input_dimensions):
            xi = jnp.squeeze(x1[:, i])
            xj = jnp.squeeze(x2[:, i])
            mesh = jnp.meshgrid(xi, xj, indexing='xy')
            xx = xx.at[i].set(mesh[0].T)
            yy = yy.at[i].set(mesh[1].T)

    l = parameters['l']
    A = parameters['A']
    g = parameters['g']
    l1 = parameters['l1']
    l2 = parameters['l2']
    k00 = A*(g**2*l**4 - 2*g*l**2*l2*(l - (xx[0] - yy[0])**2) + l2**2*(2*l**2 - 4*l*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)**2))*jnp.exp(-1/2*(xx[0] - yy[0])**2/l)/l**4
    k01 = A*(g**2*l**4 - g*l**2*(l - (xx[0] - yy[0])**2)*(l1 + l2) + l1*l2*(2*l**2 - 4*l*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)**2))*jnp.exp(-1/2*(xx[0] - yy[0])**2/l)/l**4
    k02 = A*(g**3*l**6 - g**2*l**4*(l - (xx[0] - yy[0])**2)*(l1 + 2*l2) + g*l**2*l2*(l1*(2*l**2 - 4*l*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)**2) + l1*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4) + l2*(2*l**2 - 4*l*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)**2)) - l1*l2**2*(12*l**3 - 12*l**2*(xx[0] - yy[0])**2 + 8*l*(-3*l + (xx[0] - yy[0])**2)*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4)))*jnp.exp(-1/2*(xx[0] - yy[0])**2/l)/l**6
    k10 = A*(g**2*l**4 - g*l**2*(l - (xx[0] - yy[0])**2)*(l1 + l2) + l1*l2*(2*l**2 - 4*l*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)**2))*jnp.exp(-1/2*(xx[0] - yy[0])**2/l)/l**4
    k11 = A*(g**2*l**4 - 2*g*l**2*l1*(l - (xx[0] - yy[0])**2) + l1**2*(2*l**2 - 4*l*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)**2))*jnp.exp(-1/2*(xx[0] - yy[0])**2/l)/l**4
    k12 = A*(g**3*l**6 - g**2*l**4*(l - (xx[0] - yy[0])**2)*(2*l1 + l2) + g*l**2*l1*(l1*(2*l**2 - 4*l*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)**2) + l2*(2*l**2 - 4*l*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)**2) + l2*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4)) - l1**2*l2*(12*l**3 - 12*l**2*(xx[0] - yy[0])**2 + 8*l*(-3*l + (xx[0] - yy[0])**2)*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4)))*jnp.exp(-1/2*(xx[0] - yy[0])**2/l)/l**6
    k20 = A*(g**3*l**6 - g**2*l**4*(l - (xx[0] - yy[0])**2)*(l1 + 2*l2) + g*l**2*l2*(l1*(2*l**2 - 4*l*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)**2) + l1*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4) + l2*(2*l**2 - 4*l*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)**2)) - l1*l2**2*(12*l**3 - 12*l**2*(xx[0] - yy[0])**2 + 8*l*(-3*l + (xx[0] - yy[0])**2)*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4)))*jnp.exp(-1/2*(xx[0] - yy[0])**2/l)/l**6
    k21 = A*(g**3*l**6 - g**2*l**4*(l - (xx[0] - yy[0])**2)*(2*l1 + l2) + g*l**2*l1*(l1*(2*l**2 - 4*l*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)**2) + l2*(2*l**2 - 4*l*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)**2) + l2*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4)) - l1**2*l2*(12*l**3 - 12*l**2*(xx[0] - yy[0])**2 + 8*l*(-3*l + (xx[0] - yy[0])**2)*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4)))*jnp.exp(-1/2*(xx[0] - yy[0])**2/l)/l**6
    k22 = A*(g**4*l**8 - 2*g**3*l**6*(l - (xx[0] - yy[0])**2)*(l1 + l2) + g**2*l**4*(l1**2*(2*l**2 - 4*l*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)**2) + 2*l1*l2*(2*l**2 - 4*l*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)**2) + 2*l1*l2*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4) + l2**2*(2*l**2 - 4*l*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)**2)) - 2*g*l**2*l1*l2*(l1 + l2)*(12*l**3 - 12*l**2*(xx[0] - yy[0])**2 + 8*l*(-3*l + (xx[0] - yy[0])**2)*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4)) + l1**2*l2**2*(24*l**4 - 96*l**3*(xx[0] - yy[0])**2 + 72*l**2*(l - (xx[0] - yy[0])**2)**2 - 16*l*(3*l - (xx[0] - yy[0])**2)**2*(xx[0] - yy[0])**2 + (3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4)**2))*jnp.exp(-1/2*(xx[0] - yy[0])**2/l)/l**8
    k = jnp.squeeze(jnp.concatenate([
    jnp.concatenate([k00, k01, k02], axis = -1),
    jnp.concatenate([k10, k11, k12], axis = -1),
    jnp.concatenate([k20, k21, k22], axis = -1)], axis=-2))
                           
    if include_noise:
        k += noise * jnp.eye(x1.shape[0]*num_tasks)
    return k


def model(X, Y, sigma = 5e-2, priors = None): 
    num_tasks = 3
    sampled_parameters = {}
    if priors is None: 
       priors = {}
       priors['l'] = dist.Uniform(0.1, 15)
       priors['A'] = dist.Uniform(0.1, 15)
       priors['g'] = dist.Uniform(0.1, 15)
       priors['l1'] = dist.Uniform(0.5, 4.)
       priors['l2'] = dist.Uniform(0.5, 4.)
    for key in priors:
       sampled_parameters[key] = numpyro.sample(key, priors[key])
    k = kernel(X, X, sampled_parameters, sigma = sigma)
    numpyro.sample(
        "Y",
        dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]*num_tasks), covariance_matrix=k),
        obs=Y)
                           

def run_inference(model, rng_key, X, Y, num_samples, sigma, init_values, fixed_params = {}, priors = None): #maybe make init values a kwarg
    start = time.time()
    init_strategy = init_to_value(
            values=init_values) 
    model_with_opt_fixed_params = condition(model, data=fixed_params)
    kernel = NUTS(model_with_opt_fixed_params, init_strategy=init_strategy)
    mcmc = MCMC(
        kernel,
        num_warmup=1000,
        num_samples= num_samples,
        num_chains=1,
        thinning=1,
        progress_bar=True, 
        )
    mcmc.run(rng_key, X, Y, sigma = sigma, priors = priors)
    mcmc.print_summary()
    print("MCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()                           