from document_chunk import Chunk
import get_vector
import pickle
import numpy as np
import json
from whoosh.index import open_dir
import jieba
from whoosh.qparser import QueryParser
from whoosh.query import Or
import torch
from  transformers  import AutoModelForSequenceClassification, AutoTokenizer,AutoModelForCausalLM
from tqdm import tqdm
import time 
import heapq
from sklearn.cluster import KMeans
import torch

# path = '../models/mxbai'
# tokenizer = AutoTokenizer.from_pretrained(path)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
# model.eval()

import gc
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def cal_reranker_score(data):
    path = '../models/mxbai'
    from sentence_transformers import CrossEncoder
    # 初始化模型
    from mxbai_rerank import MxbaiRerankV2
    model = MxbaiRerankV2(path)
    #model = CrossEncoder(path, max_length=512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(path)
    #model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
    #model.eval()

    inputs = tokenizer(data,
                       padding=True,
                       truncation=True,
                       return_tensors='pt',
                       max_length=100).to(device)

    with torch.no_grad():                     # 防止计算图累积
        logits = model(**inputs).logits.view(-1)
        print(logits)
        scores = torch.sigmoid(logits.float()).cpu()   # 先把结果搬到 CPU
        print(scores)

    # 1. 删除模型和张量
    del model, inputs, logits
    gc.collect()                              # 2. 触发 Python 垃圾回收
    torch.cuda.empty_cache()                  # 3. 清空 PyTorch 显存缓存
    return scores

def get_top_query(data,topK=10):
    scores=cal_reranker_score(data)
    result=sorted(list(zip(data,scores)),key=lambda s:s[1])
    result=[s[0][1] for s in result]
    return result[0:topK]

def cal_cos(vector1,vector2):
    return sum([ s1*s2 for s1,s2 in zip(vector1,vector2)])
def cal_score_w(vector,vector2s):
    if len(vector2s)==0:
        return 1.0
    score2=[ vector@vector2  for vector2 in vector2s]
    #print(len(vector2s),score2)
 
    return max(score2)
def MMR(id_scores,topK,vectors,a=0.6):
    result=[]
    vectors=np.array(vectors)
    for i in range(0,len(id_scores)):
        id,score=id_scores[i]
        vector=vectors[i]
        vector2s=vectors[0:i]
        score2=cal_score_w(vector,vector2s)
        result.append([id,a*score+(1-a)*score2])
    result=sorted(result,key=lambda s:s[1],reverse=True)[0:topK]
    return [s[0] for s in result]
def reranker_pair(data,ids,topK,vectors):
    print('start rerankerp pair')
    batch_size = 32 # 根据显存调整
    scores = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_scores = cal_reranker_score(batch)
        torch.cuda.empty_cache()
        scores.extend(batch_scores)
    ids_scores= list(zip(ids,scores)) 
    ids_vectors=dict(list(zip(ids,vectors)))
    ids_scores=heapq.nlargest(2*topK,ids_scores,key=lambda s:s[1])
    vectors=[ ids_vectors[id] for id,_ in ids_scores]
    print('start MMr')
    result=MMR(ids_scores,topK,vectors) 
    return result 
def reranker(query,ids,topK=20):
    print('start reranker')
    data=[[query,index_chunk[id].abstract] for id in ids]
    #获取所对应的向量，bert抽取的，和召回的向量是一回事
    vectors=[chunkid_vector[str(id)] for id in ids]
    result=reranker_pair(data,ids,topK,vectors)
    return result

 
 

# 打开索引
ix = open_dir("index_data/tfidf")
def get_faiss_candidate(query,num=50):
    vector=get_vector.get_vector(query)
    vector=np.array([vector])
    D, I = faiss_index.search(vector, num)
    D=D[0]
    I=I[0]
    indexs=[]
    for d,i in zip(D,I):
        indexs.append(faissid_chunkid[str(i)])
    return indexs
def get_inverted_candidate(query,num=50):
    querys=list([s for s in jieba.cut(query)])
    with ix.searcher() as searcher:
        query_list=[]
        for q in querys:
            query_list.append(QueryParser("content", ix.schema).parse(q))
        query=Or(query_list)
        result = searcher.search(query,limit=num)
        result=[int(s["path"]) for s in result]

    return result
with open("index_data/kg_index/entity_chunks",encoding="utf-8") as f:
    entity_chunkid=json.load(f)

with open("index_data/chunk_index/faissid_chunkid",encoding="utf-8") as f:
    faissid_chunkid=json.load(f)
with open("index_data/chunk_index/chunkid_vector",encoding="utf-8") as f:
    chunkid_vector=json.load(f)

with open("index_data/chunk_index/chunk_vector","rb") as f:
    faiss_index=pickle.load(f)
with open("index_data/chunk_index/index_chunk","rb") as f:
    index_chunk=pickle.load(f)
def search(query,topK):
    #向量召回
    candidate1=get_faiss_candidate(query,num=100)
    #关键词召回
    candidate2=get_inverted_candidate(query,num=100)
    #candidate3=get_kg_candidate(query)
    #ids=[ s for s in candidate1 if int(s) in set(candidate2) and int(s) in candidate3]
    #合并结果
    ids=list(set(candidate1+candidate2))
    print(ids)
    #进行排序
    ids=reranker(query,ids,topK=topK)
    #ids返回的chunkid
    contents=["知识点{}:".format(i+1)+index_chunk[s].content for i,s in enumerate(ids)]
    return "".join(contents),ids

#标准的rag
def chat(query,search_topK):
    content,ids =search(query,search_topK)
    print(content,ids)
    prompt="{}根据上述内容回答问题:{},尽可能全面".format(content,query)
    response=llm('../models/Qwen3-8B',prompt)
    return response,ids
def get_cluster_centenr(vectors,querys,n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    # 拟合模型
    kmeans.fit(np.array(vectors))
    # 获取聚类标签
    labels = kmeans.labels_
    # 获取聚类中心
    centroids = kmeans.cluster_centers_ 
    result=[]
    for i,vector in enumerate(vectors):
        label=labels[i]
        d=np.sqrt(np.sum((vector - centroids[label]) ** 2))
        result.append([querys[i],d])
    result=sorted(result,key=lambda s:s[1])
    result=[s[0] for s in result]
    return result
def llm(model_name,query,temperature=1):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        # device_map={'cuda':0}
        device_map="auto"
    )

    # prepare the model input
    messages = [
        {"role": "user", "content": f"{query}"}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    print('==========================================')
    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768,
        temperature=temperature
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    #print('问题:', query)
    #print("回答文本:", content)
    return  content

#排序模型
if __name__=="__main__":
    path='../models/Qwen3-8B'
    tokenizer = AutoTokenizer.from_pretrained(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
    model.eval()



