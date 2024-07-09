"""
RSG model simulation
"""
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.integrate import quad
from scipy.optimize import minimize

# def r(S):
#     return 10 + 50/(1+np.exp(-(S-0.25)/(0.15)))
# x = [i/100 for i in range(0, 101)]
# x *= 1
# # m = [np.random.normal(r(s), 5) for s in x ]
# m = [np.exp(-(r(s)-58)**2/(50))/(5*np.sqrt(2*np.pi)) for s in x ]

# # the plot
# plt.figure(figsize=(10, 6))  # Specify the size of the figure
# plt.scatter(x, m, marker='o', s= 10, color='k', label='Data')  # Line plot with square markers
# plt.title('Sample Plot of x vs m') 
# plt.xlabel('x-axis label')
# plt.ylabel('m-axis label')
# plt.grid(True)
# plt.legend()
# plt.show()
# print(x[m.index(max(m))])

###FORWARD###

# Define functions for estimators
def MLE_estimator(t_m, w_m, t_min=None, t_max=None):
    return t_m * ((-1 + np.sqrt(1 + 4 * w_m**2)) / (2 * w_m**2))

def MAP_estimator(t_m, w_m, t_min, t_max):
    if t_m < t_min:
        return t_min
    elif t_m > t_max:
        return t_max
    else:
        return MLE_estimator(t_m, w_m)

def prior_t_m(t_s, t_m, w_m):
    return np.exp(-(t_s - t_m)**2 / (2 * (w_m * t_s)**2)) / np.sqrt(2 * np.pi * (w_m * t_s)**2)

def BLS_integrand(t_s, t_m, w_m):
    return t_s * prior_t_m(t_s, t_m, w_m)

def BLS_estimator(t_m, w_m, t_min, t_max):
    nom, _ = quad(BLS_integrand, t_min, t_max, args=(t_m, w_m))
    denom, _ = quad(prior_t_m, t_min, t_max, args=(t_m, w_m))
    if denom == 0:
        return t_m
    return nom / denom

def model(estimator):
    if estimator == 'mle':
        return MLE_estimator
    elif estimator == 'map':
        return MAP_estimator
    elif estimator == 'bls':
        return BLS_estimator
    else:
        raise ValueError('Invalid estimator')

# Main simulation parameters
w_m = 0.1204
w_p = 0.0583
repeat_per_interval = 100
conditions = ['short', 'intermediate', 'long']
estimators = ['mle', 'map', 'bls']

def simulator(estimator, condition, w_m, w_p, repeat):
    # Define t_min and t_max based on condition
    if condition == 's' or condition == 'short':
        t_min = 494
        t_max = 847
    elif condition == 'i' or condition == 'intermediate':
        t_min = 671
        t_max = 1023
    elif condition == 'l' or condition == 'long':
        t_min = 847
        t_max = 1200
    else:
        raise ValueError('Invalid condition')
    
    estimator = model(estimator)
    
    # Generate t_s
    sample_intervals = list(range(t_min, t_max, (t_max - t_min) // 10))
    t_s = [s for s in sample_intervals for _ in range(repeat)]

    # Generate t_m based on Gaussian distribution
    t_m = [np.random.normal(x, w_m * x) for x in t_s]

    # Estimate t_e using different methods
    t_e = [estimator(x, w_m, t_min, t_max) for x in t_m]

    # Generate t_p based on Gaussian distribution
    t_p = [int(np.random.normal(x, w_p * x)) for x in t_e]

    # Calculate mean for each method
    mean_t_p = [np.average(t_p[i:i + repeat]) for i in range(0, len(t_p), repeat)]

    return sample_intervals, t_s, t_p, mean_t_p

# Simulation and plotting
def color(condition):
    if condition == 'short':
        return 'black'
    elif condition == 'intermediate':
        return 'darkred'
    elif condition == 'long':
        return 'red'
    else:
        raise ValueError('Invalid color condition')

def plot_simulation(estimators=estimators, conditions=conditions, w_m=w_m, w_p=w_p, repeat_per_interval=repeat_per_interval):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for estimator in estimators:
        for condition in conditions:
            sample_intervals, t_s, t_p, mean_t_p = simulator(estimator, condition, w_m, w_p, repeat_per_interval)
            ax = axes[estimators.index(estimator)]
            ax.scatter(t_s, t_p, marker='o', color=color(condition),s=3, alpha=0.3)
            ax.plot(sample_intervals, mean_t_p, marker='o', markersize=7, color=color(condition), label=f'{condition}')
            ax.set_xlim([400, 1300])
            ax.set_ylim([400, 1300])
            ax.set_title(f'{estimator.upper()} Estimator')
            ax.set_xlabel('t_s')
            ax.set_ylabel('t_p')
            ax.grid(True)
            ax.legend()
    plt.show()


###BACKWARD###

# Define the Probability Distribution
def P_t_m(t_s, t_m, w_m):
    return np.exp(-(t_s - t_m)**2 / (2 * (w_m * t_s)**2)) / np.sqrt(2 * np.pi * (w_m * t_s)**2)

def P_t_p(F_t_m, t_p, w_p):
    return np.exp(-(F_t_m - t_p)**2 / (2 * (w_p * F_t_m)**2)) / np.sqrt(2 * np.pi * (w_p * F_t_m)**2)

def trial_integrand(t_m, t_s, t_p, w_m, w_p, estimator, t_min, t_max):
    return P_t_m(t_s, t_m, w_m) * P_t_p(estimator(t_m, w_m, t_min, t_max), t_p, w_p)

def trial_probability(t_s, t_p, w_m, w_p, estimator, t_min, t_max):
    result, _ = quad(trial_integrand, 0, 2000, args=(t_s, t_p, w_m, w_p, estimator, t_min, t_max))
    # print(np.log(result))
    return result

# Define the Likelihood Function
def likelihood(params, t_s, t_p, estimator, t_min, t_max):
    print(f'Calculating likelihood: {params}')
    w_m, w_p = params
    # Log Likelihood
    likelihoods = [trial_probability(t_s[i], t_p[i], w_m, w_p, estimator, t_min, t_max) for i in range(len(t_s))]
    print(-np.sum(np.log(likelihoods)))
    return -np.sum(np.log(likelihoods))
    # RMSE
    # errors = [(t_p[i] - estimator(t_s[i], w_m, t_min, t_max))**2 for i in range(len(t_s))]
    # print(np.sqrt(np.mean(errors)))
    # return np.sqrt(np.mean(errors))

# Optimize the Likelihood Function
def optimize_likelihood(t_s, t_p, estimator, condition):
    print(f'Optimizing {estimator} estimator for {condition} condition')
    if condition == 's' or condition == 'short':
        t_min = 494
        t_max = 847
    elif condition == 'i' or condition == 'intermediate':
        t_min = 671
        t_max = 1023
    elif condition == 'l' or condition == 'long':
        t_min = 847
        t_max = 1200
    else:
        raise ValueError('Invalid condition')
    
    estimator = model(estimator)
    initial_w = [random.uniform(0, 0.2), random.uniform(0, 0.2)]
    result = minimize(likelihood, initial_w, args=(t_s, t_p, estimator, t_min, t_max), bounds=[(0.001, 0.3), (0.001, 0.3)])
    return result.x

# Example Usage
if __name__ == '__main__':
    # Define the simulation parameters
    w_m = 0.1208
    w_p = 0.0583
    repeat_per_interval = 10

    # Create data
    sample_intervals, t_s, t_p, mean_t_p = simulator('bls', 's', w_m, w_p, repeat_per_interval)

    # Optimize likelihood
    # likelihood([w_m, w_p],t_s, t_p, BLS_estimator, t_min = 494, t_max = 847)
    # w_m_opt, w_p_opt = optimize_likelihood(t_s, t_p, 'bls', 's')
    # print(f'Optimized w_m: {w_m_opt}, Optimized w_p: {w_p_opt}')
    
    # Error plane plot
    w_m_range = np.linspace(0.001, 0.3, 10)
    w_p_range = np.linspace(0.001, 0.3, 10)
    errors = np.zeros((100, 100))
    for i, w_m in enumerate(w_m_range):
        for j, w_p in enumerate(w_p_range):
            errors[i, j] = likelihood([w_m, w_p], t_s, t_p, BLS_estimator, t_min = 494, t_max = 847)
    plt.figure(figsize=(10, 6))
    plt.contourf(w_m_range, w_p_range, errors, levels=20)
    plt.colorbar()
    plt.title('Error Plane')
    plt.xlabel('w_m')
    plt.ylabel('w_p')
    plt.show()


    # plot_simulation()