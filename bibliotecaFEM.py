import numpy as np

def phi(i,x,nodVec):
    n=len(nodVec)-1
    if i==0:
        def tramo1(x):
            return (nodVec[1]-x)/(nodVec[1]-nodVec[0])
        return np.piecewise(x,[(nodVec[0]<=x) & (x<=nodVec[1])],[lambda x:tramo1(x)])
    elif i==n:
        def tramo1(x):
            return (x-nodVec[n-1])/(nodVec[n]-nodVec[n-1])
        return np.piecewise(x,[(nodVec[n-1]<=x) & (x<=nodVec[n])],[lambda x:tramo1(x)])
    else:
        def tramo1(x):
            return (x-nodVec[i-1])/(nodVec[i]-nodVec[i-1])
        def tramo2(x):
            return (nodVec[i+1]-x)/(nodVec[i+1]-(nodVec[i]))
        return np.piecewise(x,[(nodVec[i-1]<=x) & (x<=nodVec[i]),(nodVec[i]<=x) & (x<=nodVec[i+1])],[lambda x:tramo1(x),lambda x: tramo2(x)])


def dphi(i,x,nodVec):
  n=len(nodVec)-1
  if i==0:
    def tramo(x):
      return -1/(nodVec[1]-nodVec[0])
    return np.piecewise(x,[(nodVec[0]<=x) & (x<=nodVec[1])],[lambda x:tramo(x)])
  elif i==n:
    def tramo(x):
      return 1/(nodVec[n]-nodVec[n-1])
    return np.piecewise(x,[(nodVec[n-1]<=x) & (x<=nodVec[n])],[lambda x:tramo(x)])
  else:
    def tramo1(x):
            return 1/(nodVec[i]-nodVec[i-1])
    def tramo2(x):
            return -1/(nodVec[i+1]-(nodVec[i]))
    return np.piecewise(x,[(nodVec[i-1]<=x) & (x<=nodVec[i]),(nodVec[i]<=x) & (x<=nodVec[i+1])],[lambda x:tramo1(x),lambda x: tramo2(x)])