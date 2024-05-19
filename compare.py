import numpy as np
import matplotlib.pyplot as plt

from data.data import make_data
from model.flatl2 import FlatL2
from model.flatip import FlatIP
from model.flatl2_pca import FlatL2PCA
from model.ivf import IVF
from model.ivfpq import IVFPQ
from model.hnswflat import HNSWFlat

def main():
  # モデル定義
  k = 4
  flatL2 = FlatL2(k = k)
  flatIP = FlatIP(k = k)
  flatL2pca = FlatL2PCA(k = k, d_new = 8)
  ivf = IVF(k = k, nlist = 100, nprobe = 3)
  ivfpq = IVFPQ(k = k, nlist = 100, m = 8)
  hnswFlat = HNSWFlat(k = k, m = 20)
  models = [flatL2, flatIP, flatL2pca, ivf, ivfpq, hnswFlat]
  
  nb_list = np.logspace(4, 6, 20).astype("int")
  nq_list = [1000]
  d_list = [64]
  n_trial = 5
  time_train_list = [[[] for i in range(len(models))] for i in range(n_trial)]
  time_add_list = [[[] for i in range(len(models))] for i in range(n_trial)]
  time_search_list = [[[] for i in range(len(models))] for i in range(n_trial)]

  for n_try in range(n_trial):
    print(f"----------iteration {n_try} / {n_trial} -----------")
    for idx_nb, nb in enumerate(nb_list):
      for nq in nq_list:
        for d in d_list:
          print(f"process {idx_nb} / {len(nb_list)}")
          xb, xq, d = make_data(nb, nq, d)
          for idx, model in enumerate(models):
            time_train, time_add = model.train(xb, d)
            time_train_list[n_try][idx].append(time_train)
            time_add_list[n_try][idx].append(time_add)
            _, _, time_search = model.search(xq)
            time_search_list[n_try][idx].append(time_search)
  
  # 描画
  time_train_list = np.mean(np.array(time_train_list), axis=(0))
  time_add_list = np.mean(np.array(time_add_list), axis=(0))
  time_search_list = np.mean(np.array(time_search_list), axis=(0))
  name_models = '_'.join(model.name for model in models)
  plt.figure()
  for idx, model in enumerate(models):
    plt.plot(nb_list, time_train_list[idx], label=model.name)
  plt.xlabel("nb")
  plt.ylabel("train time [s]")
  plt.title("nb - train_time")
  plt.legend()
  plt.savefig(f"data/fig/compare_nb_train_time_{name_models}.png")
  plt.show()

  plt.figure()
  for idx, model in enumerate(models):
    plt.plot(nb_list, time_add_list[idx], label=model.name)
  plt.xlabel("nb")
  plt.ylabel("add time [s]")
  plt.title("nb - add_time")
  plt.legend()
  plt.savefig(f"data/fig/compare_nb_add_time_{name_models}.png")
  plt.show()

  plt.figure()
  for idx, model in enumerate(models):
    plt.plot(nb_list, time_search_list[idx], label=model.name)
  plt.xlabel("nb")
  plt.ylabel("search time [s]")
  plt.title("nb - search_time")
  plt.legend()
  plt.savefig(f"data/fig/compare_nb_search_time_{name_models}.png")
  plt.show()

  


if __name__=="__main__":
  main()
