import json
import faiss
import numpy as np
import pickle
import os
from openai import OpenAI

import torch
from tqdm import tqdm
 
import pickle
 
from  document_chunk import Chunk
import re
import json
from document_chunk import Chunk
from chat import chat
def split_sentences(text):
    pattern = r'([.!?。！？])'
    parts = re.split(pattern, text)
    sentences = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts):
            sentences.append(parts[i] + parts[i + 1])
            i += 2
        else:
            if parts[i].strip():
                sentences.append(parts[i])
            i += 1
    return sentences
def get_parent_chunk(chunk_info:list[Chunk],id_data,width=2):
    chunk_num=len(chunk_info)
    for i in range(width,chunk_num-width):
        ids=list(set([ k  for  j in range(i-width,i+width+1) for k in chunk_info[j].id_list]))
        ids=sorted(ids)
        parent=""
        for id in ids:
            parent+=id_data[id]
        chunk_info[i].parent=parent
    return  chunk_info

def get_chunk(data,chunk_size=256,common_token=30):
    #数据进行分句子
    data=[ ss for s in data for ss in split_sentences(s)]
    #原始数据，句子id
    id_data=dict([ [i,s] for i ,s in enumerate(data)])
    with open("index_data/chunk_index/id_data","w",encoding="utf-8") as f:
        json.dump(id_data,f,ensure_ascii=False)
    para=""
    chunk_info={}
    chunk_id=0
    id_list=[]
    id=0
    while True:
        if len(para)>=chunk_size or id >len(id_data):
            #通过大模型提取摘要
            abstract=chat("{}。提取上述句子的简短摘要，不超过50字".format(para))
            print (chunk_id,abstract)
            chunk=Chunk(para,chunk_id,None,abstract,id_list)
            chunk_info[chunk_id]=chunk
            token=""
            chunk_id+=1
            #回退common_token个字符，保证连续两个chunk之间有交集
            if len(id_data[id_list[-1]])<chunk_size:
                for i in reversed(id_list):
                    token+=id_data[i]
                    if len(token)>=common_token:
                        id=i
                        break        
            para=""
            id_list=[]
        if id>=len(id_data):
            break
        para+=id_data[id]
        id_list.append(id)
        id+=1
    #抽取chunk的父块
    chunk_info=get_parent_chunk(chunk_info,id_data)
    return chunk_info


path="明朝那些事儿.txt" 
#对文章进行切片
with open(path,encoding="utf-8") as f:
    lines=f.readlines()
data=[line.strip() for line in lines if len(line.strip())>0]
index_chunk=get_chunk(data)
with open("index_data/chunk_index/index_chunk","wb") as f:
    pickle.dump(index_chunk,f)

 