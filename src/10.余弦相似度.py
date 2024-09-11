import numpy as np


# 余弦相似度:
#   Consine Similarity = (A·B) / (||A|| * ||B||)
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)        # 点积
    magnitude1 = np.linalg.norm(vec1)       # magnitude: 范数, 模长
    magnitude2 = np.linalg.norm(vec2)
    
    return dot_product / (magnitude1 * magnitude2)


if __name__ == '__main__':
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([2, 3, 4])
    print(cosine_similarity(vec1, vec2))    # 0.9925833339709303