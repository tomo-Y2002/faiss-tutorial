import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from model.ivf import IVF
from data.data import xb, xq, d

def main():
  nlists = np.logspace(1, 3, 10).astype("int")
  time_train_list = []
  time_search_list = []
  for idx, nlist in enumerate(nlists):
    print(f"idx : {idx}, nlist : {nlist}--------")
    obj = IVF(k=4, nlist=int(nlist), nprobe=10)
    time_train = obj.train(xb, d)
    time_train_list.append(time_train)
    _, _, time_search = obj.search(xq)
    time_search_list.append(time_search)

  # 描画
  plt.figure()
  plt.plot(nlists, time_train_list, label="train")
  plt.legend()
  plt.xlabel("nlist")
  plt.ylabel("train time [s]")
  plt.title("nlist - train_time")
  plt.savefig("../data/fig/ivf_nlist_train_time.png")
  plt.show()

  plt.figure()
  plt.plot(nlists, time_search_list, label="search")
  plt.xlabel("nlist")
  plt.ylabel("process time [s]")
  plt.title("nlist - process_time")
  plt.savefig("../data/fig/ivf_nlist_search_time.png")
  plt.show()

if __name__=="__main__":
  main()