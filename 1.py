import faiss
from data import xb, xq, d

index = faiss.IndexFlatL2(d)
print(index.is_trained)
index.add(xb)
print(index.ntotal)

import time
k = 4
D, I = index.search(xb[:5], k)
print(D)  # returns top k neighbor distance
print(I)  # returns top k neighbor ID
start = time.time()
D, I = index.search(xq, k)
print(f"time : {time.time() - start}")
print(I[:5])
print(I[-5:])