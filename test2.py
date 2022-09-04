import numpy as np

def sumaCom(vec):
    #n = vec.size
    n = len(vec)
    s = 0
    for i in range(n):
        s = s + vec[i]
    return s

v = np.array([1,2,3,4,5])
suma = sumaCom(v)
print(suma)