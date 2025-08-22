from recall_rank import *
from extend_query import *
def deepsearch(origin_query):
    sub_querys=[origin_query]
    knowledge=[]
    max_num=3
    extend_query_num=3
    search_topK=50
    #目前已经获取的chunk
    exist_ids=set()
    print("========================")
    for i in range(0,max_num):
        new_ids=set()
        print (i,sub_querys)
        response_list=[]
        for query in sub_querys:
            #通过标准rag进行回答
            response,ids=chat(query,search_topK)
            print("+++++++++++++++++++++++++++++++++++")
            knowledge.append(response)
            #该query所带来的新增chunk
            new_ids.update([s for s in ids if s not in exist_ids])
            response_list.append(response)
        info_growth_rate=len(new_ids)/max(len(exist_ids),1)
        print ("第{}轮信息增长率".format(i),info_growth_rate)
        if  info_growth_rate<0.1:
            break
        exist_ids.update(new_ids)
        if i <max_num-1:
            print ("第{}轮,扩展前".format(i),sub_querys)
            sub_querys=extend_query(sub_querys,response_list,extend_query_num)
            print ("第{}轮,扩展后".format(i),sub_querys)
        #进行到下一轮深度搜索


    knowledge=["\n知识点{}:".format(i+1)+s for i,s in enumerate(knowledge)]
    knowledge="".join(knowledge)
    prompt="{}根据上述内容回答问题,尽可能全面详细:{}".format(knowledge,origin_query)
    response=llm('../models/Qwen3-8B',prompt)
    return response
if __name__ == "__main__":
    query="介绍下明朝的内阁首辅"
    result=deepsearch(query)
    print (result)