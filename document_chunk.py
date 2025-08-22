class Chunk:
    def __init__(self,content,chunk_id,parent,abstract,id_list):
        self.content=content
        self.chunk_id=chunk_id
        self.parent=parent
        self.id_list=id_list
        self.abstract=abstract