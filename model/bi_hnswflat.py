import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import faiss
from data.data import make_data
import time

class BinaryHNSW():
  def __init__(self, k, m):
    self.k = k
    self.m = m
    self.name = "Binary HNSW"

  def train(self, xb, d):
    """
    returns 
        time_train  : time elapsed during training
    """
    self.index = faiss.IndexBinaryHNSW(d * 8, self.m)

    time_train = 0
    start = time.time()
    self.index.add(xb.astype("uint8"))
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
    D, I = self.index.search(xq.astype("uint8"), self.k)
    time_search = time.time() - start
    return D, I, time_search
  
if __name__=="__main__":
  obj = BinaryHNSW(k = 4, m = 3)
  xb, xq, d = make_data()
  time_train, _ = obj.train(xb, d)
  _, I, time_search = obj.search(xq)
  print(I[-5:])
  print(f"time : {time_search}")
