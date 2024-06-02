import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import faiss
from data.data import make_data
import time

class IVF():
  def __init__(self, k, nlist, nprobe):
    self.k = k
    self.nlist = nlist
    self.nprobe = nprobe
    self.name = "IVF"
    self.order = 4

  def train(self, xb, d):
    """
    returns 
        time_train  : time elapsed during training
    """
    quantizer = faiss.IndexFlatL2(d)
    self.index = faiss.IndexIVFFlat(quantizer, d, self.nlist)
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
    self.index.nprobe = self.nprobe
    assert self.index.is_trained
    start = time.time()
    D, I = self.index.search(xq, self.k)
    time_search = time.time() - start
    return D, I, time_search
  
if __name__=="__main__":
  obj = IVF(k = 4, nlist = 1000, nprobe = 10)
  xb, xq, d = make_data()
  time_train, _ = obj.train(xb, d)
  _, I, time_search = obj.search(xq)
  print(I[-5:])
  print(f"time : {time_search}")
