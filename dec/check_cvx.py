

import numpy as np
from itertools import product
import networkx as nx
import cvxpy as cp
import math


# n = 5
# p = 5
# P = np.random.randn(n, n) * 0
# q = np.random.randn(n) * 0
# A = np.zeros((p, n))
# b = np.zeros(p)  
# G = np.zeros((p, n))
# h = np.zeros(p)

# P = P.T @ P

# P[2, 2] = 1
# q[2] = -2 * 3

# P[2, 2] = 2

# A[1, 2] = 1
# b[1] = 5

# x = cp.Variable(n)        
# prob = cp.Problem(cp.Minimize((1)*cp.quad_form(x, P) + q.T @ x ),
#                 [G @ x <= h,
#                 A @ x == b])
# # prob = cp.Problem(cp.Minimize((1)*cp.quad_form(x, P) + q.T @ x),
# #             [A @ x == b])
# # prob.solve(verbose=True)
# # prob.solve(solver=cp.ECOS, verbose=True, max_iters=500)
# prob.solve(solver=cp.ECOS, verbose=True, max_iters=500, feastol=1e-4)
# # Print result.
# print("\nThe optimal value is", (prob.value))

# for k in range(n):
#     print(x.value[k])

import cvxpy as cp

# Create two scalar optimization variables.
x = cp.Variable()


# Create two constraints.
constraints = [x  <= 5]

# Form objective.
obj = cp.Minimize((x - 3)**2)

# Form and solve problem.
prob = cp.Problem(obj, constraints)
prob.solve()

# The optimal dual variable (Lagrange multiplier) for
# a constraint is stored in constraint.dual_value.
# print("optimal (x + y == 1) dual variable", constraints[0].dual_value)
# print("optimal (x - y >= 1) dual variable", constraints[1].dual_value)
# print("x - y value:", (x - y).value)
print(x.value)

print(sum([1, 2, 3, 4]))
    
        