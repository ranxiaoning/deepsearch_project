import json
import faiss
import numpy as np
import pickle
import os


import torch
from tqdm import tqdm
 
import pickle
import get_vector
from  document_chunk import Chunk
 

with open("./index_data/chunk_index/index_chunk","rb") as f:
    index_chunk=pickle.load(f)
#对切片进行建立索引
chunkid_vector={}
count=0
faissid_chunkid={}
id_vector=[]
for s in tqdm(index_chunk.values()): 
    count+=1
    chunk_id=s.chunk_id
    #chunk对应的内容生成向量
    vector=get_vector.get_vector(s.content)
    chunkid_vector[chunk_id]=vector
    #把生成向量保存
    faissid_chunkid[len(id_vector)]=chunk_id
    id_vector.append(vector)
#bert的向量维度是768，所以声明一个faiss索引库
index = faiss.IndexFlatL2(1024)
id_vector=np.array(id_vector)
#把训练好的向量放进索引库，并保存
index.add(id_vector)    
with open("index_data/chunk_index/chunk_vector","wb") as f:
    pickle.dump(index,f)
with open("index_data/chunk_index/faissid_chunkid","w",encoding="utf-8") as f:
    json.dump(faissid_chunkid,f)
with open("index_data/chunk_index/chunkid_vector","w",encoding="utf-8") as f:
    json.dump(chunkid_vector,f)