import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import faiss
from data.data import xb, xq, d
import time

class IVF():
  def __init__(self, k, nlist, nprobe):
    self.k = k
    self.nlist = nlist
    self.nprobe = nprobe

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
    self.index.add(xb)

    return time_train
  
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
  time_train = obj.train(xb, d)
  _, I, time_search = obj.search(xq)
  print(I[-5:])
  print(f"time : {time_search}")
