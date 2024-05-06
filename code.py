import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd

# this function runs the markov chain for some steps
# requires the proposal matrix, the acceptance matrix, and the initial state
def mcmc_sample(init_state, C, A, steps=100):
    n = C.shape[0]
    s = init_state
    for t in range(steps):
        new_s = np.random.choice(n, p=C[:,s])
        prob_acceptance = A[new_s, s]
        if np.random.rand() < prob_acceptance:
            s = new_s
    return s

def mcmc_sampling(p, n_samples=1, steps=100):
    n = len(p)
    # first we construct a stochastic process that targets the given p
    # we follow the Metropolis-Hastings prescription
    
    # Move proposal 
    # we chose to move along the states as if they were organized in a 1D chain
    C = np.zeros((n,n))
    for i in range(n):
        if i < n-1: C[i+1,i] = 1
        if i > 0: C[i-1,i] = 1
    C = C / C.sum(axis=1) #"summing over the columns C has to be normalized"

    # Move acceptance (computed according to the Metropolis-Hastings formula)    
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if C[i,j] == 0:
                continue
            A[i,j] = min(1, C[j,i] * p[i] / (C[i,j] * p[j]))
    
    # prepare the array that is going to store the samples from p
    ss = np.zeros(n_samples, dtype=int)
    
    s = np.random.randint(n)
    for i in range(n_samples):
        # one could initialize each markov chain from a random intial state
        # s = np.random.randint(n)
        # but it is also possible to continue the previous chain
        # if the steps are enough the subsequent samples will decorrelate
        s = mcmc_sample(s, C, A, steps=steps)
        ss[i] = s
    
    return ss

# this function ontains the FREQ of each STATE given the sampled configuration
# to be compared with the actual probability p
def p_empirical(n_states, sample_vec):
    return np.array([(sample_vec==i).sum()/len(sample_vec) for i in range(n_states)])

#VISUALIZATION

num_circle_points = 100 
thetas = np.linspace(0, np.pi/2, num_circle_points) 

xs = np.cos(thetas) 
ys = np.sin(thetas) 

plt.clf() 
plt.plot(xs, ys, 'r')

n = 10000

pxs = rnd.rand(n)
pys = rnd.rand(n)

pi_est = 4 * ninside / n
print(f"pi_est = {pi_est}, pi = {np.pi}")

# we can even use a boolean mask to plot the points with different colors:
inside_mask = pxs**2 + pys**2 <= 1 
ninside = inside_mask.sum() 
plt.plot(pxs[inside_mask], pys[inside_mask], '.', color='r')
plt.plot(pxs[np.logical_not(inside_mask)], pys[np.logical_not(inside_mask)], '.', color='b')

# EXAMPLE
# note: if p contains zeros because of the move we chose (only move right or left in the 1D chain) 
# this will create a bottleneck: the MCMC will not be able to pass. 
p = np.array([0.1,0.2,0.3,0.05,0.35])
s = mcmc_sampling(p, n_samples=1000, steps=100)
p_emp = p_empirical(len(p), s)
print("true p:")
print(p)
print("empirical p (mcmc):")
print(p_emp)

