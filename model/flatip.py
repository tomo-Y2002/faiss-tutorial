import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import faiss
from data.data import make_data
import time

class FlatIP():
  def __init__(self, k):
    self.k = k
    self.name = "FlatIP"

  def train(self, xb, d):
    """
    returns time spend on training
    """
    self.index = faiss.IndexFlatIP(d)
    time_train = 0
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
    start = time.time()
    assert self.index.is_trained
    D, I = self.index.search(xq, self.k)
    time_search = time.time() - start
    return D, I, time_search

if __name__=="__main__":
  obj = FlatIP(k = 4)
  xb, xq, d = make_data()
  time_train, _ = obj.train(xb, d)
  D, I, _ = obj.search(xb[:5])
  print(D)
  print(I)
  _, I, time_search = obj.search(xq)
  print(I[:5])
  print(I[-5:])
  print(f"time_train : {time_train}, time_search : {time_search}")


    