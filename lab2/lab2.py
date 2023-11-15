
import numpy as np
import scipy as sp
from scipy.optimize import linprog
C = np.array([2, -1, 3,-2, 1])
negativeC = -1*C
print(negativeC)
A = [[-1,1,1,0,0],
     [1,-1,0,1,0],
     [1,1,0,0,1]
     ]
B = [1,1,2]

bounds = ((0, None), (0, None),(0, None),(0, None),(0, None))

res = linprog(negativeC, A_eq=A, b_eq=B,  bounds=bounds, method='simplex', options={"disp": True})


print(f"optimal function value: {round(res.fun*-1)}")
i = 1
values = res.get("x")
for x in np.array(values):
    print(f"x{i} = {round(x)}")
    i = i + 1

