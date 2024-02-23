import numpy as np
from pydrake.all import HPolyhedron, RandomGenerator

# Create a polyhedron which is the line between (1,1) and (3,1)
A_p = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
b_p = np.array([3, -1, 1, -1])
P = HPolyhedron(A_p, b_p)

n_samples = 10
samples = []
generator = RandomGenerator()
initial_guess = P.MaybeGetFeasiblePoint()
try:
    samples.append(P.UniformSample(generator, initial_guess))
    for i in range(n_samples - 1):
        samples.append(P.UniformSample(generator, previous_sample=samples[-1]))
except:
    print("Warning: failed to sample convex set")

print(samples)
