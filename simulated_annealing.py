import numpy as np

def metropolis_acceptance_prob(dy, temp):
    if dy <= 0:
        return np.float64(1.0)
    else:
        return np.exp(-dy/temp)
    
def generate_exponential_annealing_schedule(gamma, t_init):
    return lambda t: np.power(gamma, t) * t_init

# implements simulated annealing to minimize the function f
def simulated_annealing(f, x, T, t, k_max):
    x_current = x
    y_current = f(x_current)
    x_best, y_best = x_current, y_current
    for k in range(k_max):
        x_new = T(x)
        y_new = f(x_new)
        dy = y_new - y_current
        #todo: add way to seed RNG
        if dy <= 0 or np.random.random() < metropolis_acceptance_prob(dy, t(k)):
            x_current, y_current = x_new, y_new
        if y_new < y_best:
            x_best, y_best = x_new, y_new
    return x_best