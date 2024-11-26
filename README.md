# particles

This repository contains Python implementations and simulations related to the Fokker-Planck equation and Langevin dynamics, two fundamental mathematical frameworks for modeling stochastic processes in physics, chemistry, finance, and machine learning.

## Fokker-Planck Equation 1d Simulation 
j
See notebooks/fokker_planck_equation_1d_simulation.ipynb for a simulation of the Fokker-Planck equation in 1d.

```python
    # define negative log likelihood function
    def negative_log_likelihood(params, data, dt):
        mu, sigma = params
        returns = data['Return'].values
        N = len(returns)
        nll = 0.5 * N * np.log(2 * np.pi * sigma**2 * dt) + \
            np.sum((returns - mu * dt)**2) / (2 * sigma**2 * dt)
        return nll

    # set initial parameters and bounds
    initial_params = [0.0, 0.01]
    bounds = [(-np.inf, np.inf), (1e-6, np.inf)]

    # optimize negative log likelihood function
    result = minimize(
        negative_log_likelihood, 
        initial_params, 
        args=(data, dt), 
        bounds=bounds, 
        method='L-BFGS-B'
    )
    estimated_mu, estimated_sigma = result.x
```

<p align="center">
    <img src="./.github/assets/estimated_return_distribution_100.png">
</p>

<p align="center">
    <img src="./.github/assets/estimated_return_distribution_1000.png">
</p>

<p align="center">
    <img src="./.github/assets/estimated_return_distribution_10000.png">
</p>