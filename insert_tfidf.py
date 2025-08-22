import os.path
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
import pickle
#轻量级的elastic search
with open("index_data\\chunk_index\\index_chunk","rb") as f:
    index_chunk=pickle.load(f)
# 定义Schema
schema = Schema(content=TEXT, path=ID(stored=True))
ix = create_in("index_data\\tfidf", schema)
writer = ix.writer()
for  chunk in index_chunk.values():
    print (chunk)
    writer.add_document(content=chunk.content, path=str(chunk.chunk_id))
writer.commit()
print("索引创建完成")