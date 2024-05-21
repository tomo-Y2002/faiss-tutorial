import numpy as np
import matplotlib.pyplot as plt

from data.data import make_data
from model.flatl2 import FlatL2
from model.flatip import FlatIP
from model.flatl2_pca import FlatL2PCA
from model.ivf import IVF
from model.ivfpq import IVFPQ
from model.hnswflat import HNSWFlat
from model.bi_hnswflat import BinaryHNSW

def get_gt(xb, xq, d, k):
  """
  return : 
    gt : 2D np.ndarray, indexes for k nearest neighbors
  
  "gt" means grand truth.
  """
  flatL2 = FlatL2(k = k)
  _, _ = flatL2.train(xb, d)
  _, gt, _ = flatL2.search(xq)
  return gt

def get_recall(I, gt, n_recall):
  """
  I : 2D np.ndarray, indexes for k nearest neighbors
  gt : same shape as I, grand truth (calculated by Flat)

  return : 
    recall : how much top-n_recall IDs are similar

  ex)
  if n_recall = 1, this function returns recall@1
  if n_recall = 10, this function returns recall@10
  """
  assert I.shape[0] == gt.shape[0]
  assert I.shape[1] >= n_recall

  nq = I.shape[0]
  recall = 0
  for i in range(nq):
    recall += len(set(I[i, :n_recall]) & set(gt[i, :n_recall])) / float(n_recall)
  recall /= nq
  return recall

def plot_qps_hnsw():
  k = 4
  
  nb = 10**6
  nq = 1000
  d = 64
  n_trial = 5
  n_recall = 1
  efs = [2**i for i in range(2, 11)]
  qps_list = [[] for i in range(n_trial)]
  recall_list = [[] for i in range(n_trial)]
  

  for n_try in range(n_trial):
    print(f"----------iteration {n_try} / {n_trial} -----------")
    xb, xq, d = make_data(nb, nq, d)
    gt = get_gt(xb=xb, xq=xq, d=d, k=k)

    for ef in efs:
      print(f"ef = {ef}")

      model = HNSWFlat(k = k, m = ef)
      # train
      _, _ = model.train(xb, d)
      print("train done")

      # search
      _, I, time_search = model.search(xq)
      print("search done")

      # calculate QPS, Recall
      qps_list[n_try].append(nq / time_search)
      recall_list[n_try].append(get_recall(I=I, gt=gt, n_recall=n_recall))

  recall_list = np.mean(np.array(recall_list), axis=(0))
  qps_list = np.mean(np.array(qps_list), axis=(0))
  print(f"recall_list \n {recall_list}")
  print(f"qps_list \n {qps_list}")

  #描画
  name_models = "HNSW"

  plt.figure()
  
  plt.plot(recall_list, qps_list, label=model.name)
  plt.xlabel(f"Recall@{n_recall}")
  plt.ylabel("Query Per Second [s]")
  plt.yscale("log")
  plt.title("recall - pqs")

  for i, ef in enumerate(efs):
        plt.annotate(f"ef = {ef}", (recall_list[i], qps_list[i]), textcoords="offset points", xytext=(5,10), ha='center')

  plt.legend()
  plt.savefig(f"data/fig/compare_recall_qps_{name_models}.png")
  plt.show()


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
  models = [flatL2, flatIP, flatL2pca, ivf, ivfpq, hnswFlat, bi_hnsw]
  # models = [hnswFlat]
  
  # nb_list = np.logspace(4, 6, 10).astype("int")
  nb_list = np.linspace(10**4, 10**6, 20).astype("int")
  nq_list = [1000]
  d_list = [64]
  n_trial = 1
  n_recall = 1
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
            # train
            time_train, time_add = model.train(xb, d)
            time_train_list[n_try][idx].append(time_train)
            time_add_list[n_try][idx].append(time_add)
            # search
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
  plot_qps_hnsw()
