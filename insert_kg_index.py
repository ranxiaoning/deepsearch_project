from langchain_openai import ChatOpenAI
import json
from tqdm import tqdm
from chat import chat

entity_prompt="抽取上述句子的三元组，实体为人，地点，历史事件，主谓宾用-隔开，如刘备-兄弟-关羽，三元组之间用|隔开"
def extract_entity(data):
    entity_chunk={}
    for s in tqdm(data):
        if "parent" not in s:
            continue
        #对chunk的父块进行抽取三元组
        #也可以用content
        text=s['parent']
        triples=chat(text+entity_prompt)
        triples=set(triples.split("|"))
        print (triples)
        for triple in triples:
            if triple not in entity_chunk:
                entity_chunk[triple]=[]
            entity_chunk[triple].append(s["chunk_id"])
        with open("index_data\\kg_index\\entity","w",encoding="utf-8") as f:
            json.dump(entity_chunk,f,ensure_ascii=False)


    return entity_chunk
with open("index_data\\chunk_index\\index_chunk",encoding="utf-8")as f:
    lines=[eval(s.strip()) for s in f.readlines()]
entity_chunk=extract_entity(lines)
with open("index_data\\kg_index\\entity","w",encoding="utf-8") as f:
    json.dump(entity_chunk,f,ensure_ascii=False)
