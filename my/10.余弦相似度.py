import numpy as np

# 余弦相似度
#   Consine Similarity = (A * B) / ||A|| * ||B||
def cos_sim(vce1, vec2):
    dot_product = np.dot(vce1, vec2)
    magnitude1 = np.linalg.norm(vce1)
    magnitude2 = np.linalg.norm(vec2)

    return dot_product / magnitude1 * magnitude2


if __name__ == '__main__':
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([2, 3, 4])
    print(cos_sim(vec1, vec2))    # 0.9925833339709303