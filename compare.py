import numpy as np
import matplotlib.pyplot as plt

from data.data import make_data
from model.linear import FlatL2
from model.ivf import IVF

def main():
  # モデル定義
  k = 4
  flatL2 = FlatL2(k = k)
  ivf = IVF(k = k, nlist = 100, nprobe = 3)
  models = [flatL2, ivf]
  
  nb_list = np.logspace(4, 6, 10).astype("int")
  nq_list = [1000]
  d_list = [64]
  n_trial = 5
  time_train_list = [[[] for i in range(len(models))] for i in range(n_trial)]
  time_search_list = [[[] for i in range(len(models))] for i in range(n_trial)]

  for n_try in range(n_trial):
    print(f"----------iteration {n_try} / {n_trial} -----------")
    for idx_nb, nb in enumerate(nb_list):
      for nq in nq_list:
        for d in d_list:
          print(f"process {idx_nb} / {len(nb_list)}")
          xb, xq, d = make_data(nb, nq, d)
          for idx, model in enumerate(models):
            time_train = model.train(xb, d)
            time_train_list[n_try][idx].append(time_train)
            _, _, time_search = model.search(xq)
            time_search_list[n_try][idx].append(time_search)
  
  # 描画
  plt.figure()
  for idx, model in enumerate(models):
    plt.plot(nb_list, time_train_list[idx], label=model.name)
    ## ここからスタート　挑戦回数の平均を取る
  plt.xlabel("nb")
  plt.ylabel("train time [s]")
  plt.title("nb - train_time")
  plt.legend()
  plt.savefig("data/fig/compare_nb_train_time.png")
  plt.show()

  plt.figure()
  for idx, model in enumerate(models):
    plt.plot(nb_list, time_search_list[idx], label=model.name)
  plt.xlabel("nb")
  plt.ylabel("search time [s]")
  plt.title("nb - search_time")
  plt.legend()
  plt.savefig("data/fig/compare_nb_search_time.png")
  plt.show()

  


if __name__=="__main__":
  main()
