import numpy as np

a = [[0, 1, 2], [3, 4, 5]]
print(a)
print(np.shape(a))

a = np.append(a, [[6],[7]], axis = 1)

print(a)
print(np.shape(a))

