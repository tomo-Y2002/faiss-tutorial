import faiss
from data import xb, xq, d
import time
import numpy as np


nlists = np.logspace(1, 10000, 10)
time_train = []
time_process = []
for nlist in nlists:
  k = 4
  quantizer = faiss.IndexFlatL2(d)
  index = faiss.IndexIVFFlat(quantizer, d, nlist)
  assert not index.is_trained
  start = time.time()
  index.train(xb)
  time_train.append(time.time()-start)
  assert index.is_trained

  index.add(xb)
  index.nprobe = 10
  start = time.time()
  D, I = index.search(xq, k)
  time_process.append(time.time()-start)

# 描画
