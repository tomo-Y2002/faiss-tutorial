import numpy as np

def make_data(nb = 100000, nq = 10000, d = 64):
  np.random.seed(1234)
  xb = np.random.random((nb, d)).astype("float32")
  xb[:, 0] += np.arange(nb) / 1000.
  xq = np.random.random((nq, d)).astype("float32")
  xq[:, 0] += np.arange(nq) / 1000.
  return xb, xq, d