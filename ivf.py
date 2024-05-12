import faiss
from data import xb, xq, d
import time
import numpy as np
import matplotlib.pyplot as plt

nlists = np.logspace(1, 3, 10).astype("int")
time_train = []
time_process = []
for idx, nlist in enumerate(nlists):
  print(f"idx : {idx}, nlist : {nlist}--------")
  k = 4
  quantizer = faiss.IndexFlatL2(d)
  index = faiss.IndexIVFFlat(quantizer, d, int(nlist))
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
plt.figure()
plt.plot(nlists, time_train, label="train")
plt.legend()
plt.xlabel("nlist")
plt.ylabel("train time [s]")
plt.title("nlist - train_time")
plt.savefig("data/ivf_nlist_train_time.png")
plt.show()

plt.figure()
plt.plot(nlists, time_process, label="process")
plt.xlabel("nlist")
plt.ylabel("process time [s]")
plt.title("nlist - process_time")
plt.savefig("data/ivf_nlist_train_time.png")
plt.show()
