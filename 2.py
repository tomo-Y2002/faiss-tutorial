import faiss
from data import xb, xq, d
import time

nlist = 100
k = 4
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)
assert not index.is_trained
index.train(xb)
assert index.is_trained

index.add(xb)
D, I = index.search(xq, k)
print(I[-5:])
index.nprobe = 10
start = time.time()
D, I = index.search(xq, k)
print(f"time : {time.time() - start}")
print(I[-5:])