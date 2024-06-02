import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import torch

from data.data import make_data
from plot.compare_qps import plot_qps_hnsw
from model.flatl2 import FlatL2
from model.flatip import FlatIP
from model.flatl2_pca import FlatL2PCA
from model.ivf import IVF
from model.ivfpq import IVFPQ
from model.hnswflat import HNSWFlat
from model.bi_hnswflat import BinaryHNSW
from model.gpu_flatl2 import GpuFlatL2
from model.gpu_ivf import GpuIVF
from model.gpu_ivfpq import GpuIVFPQ

def gen_color(order):
  colors = plt.get_cmap("tab20").colors
  index = (order - 1) % 100 + 10 * (order - 1)//100
  return colors[index]

def main():
  # モデル定義
  k = 4
  flatL2 = FlatL2(k = k)
  flatIP = FlatIP(k = k)
  flatL2pca = FlatL2PCA(k = k, d_new = 8)
  ivf = IVF(k = k, nlist = 100, nprobe = 3)
  ivfpq = IVFPQ(k = k, nlist = 100, m = 8)
  hnswFlat = HNSWFlat(k = k, m = 32)
  bi_hnsw = BinaryHNSW(k = k, m = 32)
  if torch.cuda.is_available():
    gpuflatL2 = GpuFlatL2(k = k)
    gpuivf = GpuIVF(k = k, nlist = 100, nprobe = 3)
    gpuivfpq = GpuIVFPQ(k = k, nlist = 100, m = 8)
  # models = [flatL2, flatIP, flatL2pca, ivf, ivfpq, hnswFlat, bi_hnsw, gpuflatL2, gpuivf, gpuivfpq]
  models = [flatL2, flatIP, flatL2pca, ivf, ivfpq, hnswFlat, bi_hnsw]
  name_models = '_'.join(model.name for model in sorted(models, key = lambda x: x.order))
  
  # nb_list = np.logspace(4, 6, 10).astype("int")
  nb_list = np.linspace(10**4, 10**4.1, 2).astype("int")
  nq_list = [1000]
  d_list = [64]
  n_trial = 2
  time_train_list = np.zeros((n_trial, len(models), len(nb_list)))
  time_add_list = np.zeros((n_trial, len(models), len(nb_list)))
  time_search_list = np.zeros((n_trial, len(models), len(nb_list)))

  for n_try in range(n_trial):
    print(f"----------iteration {n_try} / {n_trial} -----------")
    for idx_nb, nb in enumerate(nb_list):
      for nq in nq_list:
        for d in d_list:
          print(f"process {idx_nb} / {len(nb_list)}")
          xb, xq, d = make_data(nb, nq, d)
          for idx, model in enumerate(models):
            # train
            time_train, time_add = model.train(xb, d)
            time_train_list[n_try][idx][idx_nb] = time_train
            time_add_list[n_try][idx][idx_nb] = time_add
            # search
            _, _, time_search = model.search(xq)
            time_search_list[n_try][idx][idx_nb] = time_search

  # pandas
  df_time_train = pd.DataFrame()
  df_time_train_list = [pd.DataFrame(time_train_list[i], columns = nb_list) for i in range(n_trial)]
  for i in range(n_trial):
    df_time_train_list[i]["trial"] = i+1
    df_time_train_list[i]["model"] = [model.name for model in models]
    df_time_train = pd.concat([df_time_train, df_time_train_list[i]], axis=0)
  df_time_train.to_csv(f"data/csv/time_train_{name_models}.csv", index = False)

  df_time_add = pd.DataFrame()
  df_time_add_list = [pd.DataFrame(time_add_list[i], columns = nb_list) for i in range(n_trial)]
  for i in range(n_trial):
    df_time_add_list[i]["trial"] = i+1
    df_time_add_list[i]["model"] = [model.name for model in models]
    df_time_add = pd.concat([df_time_add, df_time_add_list[i]], axis=0)
  df_time_add.to_csv(f"data/csv/time_add_{name_models}.csv", index = False)

  df_time_search = pd.DataFrame()
  df_time_search_list = [pd.DataFrame(time_search_list[i], columns = nb_list) for i in range(n_trial)]
  for i in range(n_trial):
    df_time_search_list[i]["trial"] = i+1
    df_time_search_list[i]["model"] = [model.name for model in models]
    df_time_search = pd.concat([df_time_search, df_time_search_list[i]], axis=0)
  df_time_search.to_csv(f"data/csv/time_search_{name_models}.csv", index = False)
            
  # 読み取り
  df_time_train_read = pd.read_csv(f"data/csv/time_train_{name_models}.csv")
  df_time_train_read = df_time_train_read.drop(columns = "trial").groupby("model").mean()
  df_time_add_read = pd.read_csv(f"data/csv/time_add_{name_models}.csv")
  df_time_add_read = df_time_add_read.drop(columns = "trial").groupby("model").mean()
  df_time_search_read = pd.read_csv(f"data/csv/time_search_{name_models}.csv")
  df_time_search_read = df_time_search_read.drop(columns = "trial").groupby("model").mean()

  # 描画
  plt.figure()
  for model in models:
    plt.plot(nb_list, df_time_train_read.loc[model.name], label=model.name, color=gen_color(model.order))
  plt.xlabel("nb")
  plt.ylabel("train time [s]")
  plt.title("nb - train_time")
  plt.legend()
  plt.savefig(f"data/fig/compare_nb_train_time_{name_models}.png")
  plt.show()

  plt.figure()
  for model in models:
    plt.plot(nb_list, df_time_add_read.loc[model.name], label=model.name, color=gen_color(model.order))
  plt.xlabel("nb")
  plt.ylabel("add time [s]")
  plt.title("nb - add_time")
  plt.legend()
  plt.savefig(f"data/fig/compare_nb_add_time_{name_models}.png")
  plt.show()

  plt.figure()
  for model in models:
    plt.plot(nb_list, df_time_search_read.loc[model.name], label=model.name, color=gen_color(model.order))
  plt.xlabel("nb")
  plt.ylabel("search time [s]")
  plt.title("nb - search_time")
  plt.legend()
  plt.savefig(f"data/fig/compare_nb_search_time_{name_models}.png")
  plt.show()


if __name__=="__main__":
  main()
  # plot_qps_hnsw()
