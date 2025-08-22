from sentence_transformers import SentenceTransformer
import torch

# 检查设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 加载 SentenceTransformer 模型
model_path = "../models/Qwen3-Eb-06.B"
model = SentenceTransformer(model_path, device=device)

# 归一化函数
def normalize(vector):
    norm = torch.norm(torch.tensor(vector, dtype=torch.float32))
    return (torch.tensor(vector) / norm).tolist()

# 获取句向量
def get_vector(sentence):
    vector = model.encode(sentence, convert_to_tensor=False)  # 返回 numpy 或 list
    return normalize(vector)
