import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import faiss
from data.data import make_data
import time

class GpuIVFPQ():
  def __init__(self, k, nlist, m):
    """
    k : top-k  search up to k th nearest neighbor
    nlist : K in PQ. number of columns in code book.
    m : M in PQ. number of rows in code book. quatinization.
    """
    self.k = k
    self.nlist = nlist
    self.m = m
    self.name = "GpuIVF-PQ"
    self.res = faiss.StandardGpuResources()
    self.config = faiss.GpuIndexIVFPQConfig()
    self.config.device = 0

  def train(self, xb, d):
    """
    returns 
        time_train  : time elapsed during training
    """
    self.index = faiss.GpuIndexIVFPQ(self.res, d, self.nlist, self.m, 8, faiss.METRIC_L2, self.config)
    assert not self.index.is_trained

    start = time.time()
    self.index.train(xb)
    time_train = time.time() - start
    assert self.index.is_trained

    start = time.time()
    self.index.add(xb)
    time_add = time.time() - start

    return time_train, time_add
  
  def search(self, xq):
    """
    returns 
        D : distance nearest k neighbor (n, k)
        I : Id nearest k neighbor (n, k)
        time_search : time spend at searchings
    """
    assert self.index.is_trained
    start = time.time()
    D, I = self.index.search(xq, self.k)
    time_search = time.time() - start
    return D, I, time_search
  
if __name__=="__main__":
  obj = GpuIVFPQ(k = 4, nlist = 100, m = 4)
  xb, xq, d = make_data()
  time_train, _ = obj.train(xb, d)
  _, I, time_search = obj.search(xq)
  print(I[-5:])
  print(f"time : {time_search}")