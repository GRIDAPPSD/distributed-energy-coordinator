

import numpy as np
from itertools import product
import networkx as nx
import cvxpy as cp
import math


nbus = 14
n  = nbus
p = 14
P = np.random.randn(n, n) * 0
q = np.random.randn(n) * 0
A = np.zeros((p, n))
b = np.zeros(p)  
G = np.zeros((p, n))
h = np.zeros(p)

#

#
for k in range(nbus):
    P[k, k] = 1
#
P[0, 0] += 5


q[0] = -10.5017329

counteq = 0
for k in range(nbus-1):
    print(k)
    A[counteq, k] = 1
    A[counteq, k+1] = -1
    b[counteq] = 0
    counteq += 1


x = cp.Variable(n)
prob = cp.Problem(cp.Minimize((1)*cp.quad_form(x, P) + q.T @ x ),
                  [A @ x == b])

# prob = cp.Problem(cp.Minimize((1)*cp.quad_form(x, P) + q.T @ x),
#
# prob.solve(verbose=True)
# prob.solve(solver=cp.ECOS, verbose=True, max_iters=500)
prob.solve(solver=cp.ECOS, verbose=True, max_iters=500, feastol=1e-4)
# Print result.
print("\nThe optimal value is", (prob.value))

for k in range(n):
    print(x.value[k])

# import cvxpy as cp

# # Create two scalar optimization variables.
# x = cp.Variable()
#
#
# # Create two constraints.
# constraints = [x  <= 5]
#
# # Form objective.
# obj = cp.Minimize(8*x**2 - 11.1486 * x +10)
#
# # Form and solve problem.
# prob = cp.Problem(obj, constraints)
# prob.solve()
#
# # The optimal dual variable (Lagrange multiplier) for
# # a constraint is stored in constraint.dual_value.
# # print("optimal (x + y == 1) dual variable", constraints[0].dual_value)
# # print("optimal (x - y >= 1) dual variable", constraints[1].dual_value)
# # print("x - y value:", (x - y).value)
# print(x.value)

    
        