import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import faiss
from data.data import xb, xq, d
import time

class FlatL2():
  def __init__(self, k):
    self.k = k

  def train(self, xb, d):
    """
    returns time spend on training
    """
    self.index = faiss.IndexFlatL2(d)
    self.index.add(xb)
    time_train = 0
    return time_train
  
  def search(self, xq):
    """
    returns 
        D : distance nearest k neighbor (n, k)
        I : Id nearest k neighbor (n, k)
        time_search : time spend at searchings
    """
    start = time.time()
    assert self.index.is_trained
    D, I = self.index.search(xq, self.k)
    time_search = time.time() - start
    return D, I, time_search

if __name__=="__main__":
  obj = FlatL2(k = 4)
  time_train = obj.train(xb, d)
  D, I, _ = obj.search(xb[:5])
  print(D)
  print(I)
  _, I, time_search = obj.search(xq)
  print(I[:5])
  print(I[-5:])
  print(f"time_train : {time_train}, time_search : {time_search}")


    