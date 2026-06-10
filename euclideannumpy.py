import numpy as np
def euclidean_distance(x,y):
  X = np.array(x)
  Y = np.array(y)
  return np.sqrt(np.sum(np.square(X - Y).item())
