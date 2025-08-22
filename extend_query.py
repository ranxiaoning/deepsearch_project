from recall_rank import *
def combine_query(q1,q2):
    prompt="将搜索词'{}'和搜索词'{}'进行合并改写成一个搜索词,只输出搜索词，不要输出无关内容".format(q1,q2)
    result=llm('../models/Qwen3-8B',prompt)
    return result
def extend_query(querys,response_list,num):
    next_querys=[]
    all_querys_scores=[]
    for query,response in zip(querys,response_list):
        if response==None:
            continue
        querys_scores=generate_subquery(query,response,num)
        all_querys_scores.extend(querys_scores)
    all_querys_scores=sorted(all_querys_scores,key=lambda s:s[-1],reverse=True)
    next_querys=[combine_query(s[0],s[1]) for s in all_querys_scores[0:num] ]
    return next_querys
def generate_subquery(query,response,num):
    prompt="{}根据上述内容，提取{}个核心搜索词，进行下一步搜索，搜索词之间用|隔开，不要输出不相关内容".format(response,2*num) 
    extend_query=llm('../models/Qwen3-8B',prompt,temperature=1.7).split("|")
    extend_query=[s for s in extend_query if len(s)>1]
    #计算改写后query和元query的得分
    scores=cal_reranker_score([[query,s] for s in extend_query])
    scores=[s.float() for s in scores]
    return list(zip(len(extend_query)*[query],extend_query,scores))

