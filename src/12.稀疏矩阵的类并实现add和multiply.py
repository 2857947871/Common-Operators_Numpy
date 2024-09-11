# 稀疏矩阵:
#   见noteability


import numpy as np

class SparseMatrix:
    def __init__(self, matrix):
        # 将稀疏矩阵转换为COO格式：[(row, col, value), ...]
        self.matrix = []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] != 0:
                    self.matrix.append((i, j, matrix[i][j]))

    def add(self, other_matrix):
        result = {(i, j): value for i, j, value in self.matrix}
        # 遍历另一个矩阵的非零元素并相加
        for i, j, value in other_matrix.matrix:
            if (i, j) in result:
                result[(i, j)] += value
            else:
                result[(i, j)]  = value

        # 将结果转换为list
        # 先进行for循环, 然后取出i, 在用max选出最大的i, 然后加1, 作为行数
        max_row = max(i for i, j, _ in self.matrix + other_matrix.matrix) + 1 # 计算结果矩阵的行数
        max_col = max(j for i, j, _ in self.matrix + other_matrix.matrix) + 1 # 计算结果矩阵的列数

        return self.dict2list(max_row, max_col, result)

    def multiply(self, other_matrix):
        self_rows = max(i for i, j, _ in self.matrix) + 1
        self_cols = max(j for i, j, _ in self.matrix) + 1
        other_rows = max(i for i, j, _ in other_matrix.matrix) + 1
        other_cols = max(j for i, j, _ in other_matrix.matrix) + 1

        if (self_cols != other_rows):
            raise ValueError("The number of columns in the first matrix must be equal to the number of rows in the second matrix")

        # 行: i, k   列: j, l
        # 行 * 列: 
        result = {}
        for i, j, value in self.matrix:
            for k, l, other_value in other_matrix.matrix:
                if k == j:
                    if (i, l) in result:
                        result[(i, l)] += value * other_value
                    else:
                        result[(i, l)]  = value * other_value

        # 生成结果矩阵
        max_row = max(i for i, _ in result) + 1
        max_col = max(j for _, j in result) + 1

        return self.dict2list(max_row, max_col, result)        

    # TODO Rename this here and in `add` and `multiply`
    def dict2list(self, max_row, max_col, result):
        
        result_matrix = [[0] * max_col for _ in range(max_row)]
        
        for (i, j), value in result.items():
            result_matrix[i][j] = value
        return SparseMatrix(result_matrix)        

    def __str__(self):
        # 打印稀疏矩阵
        max_row = max(i for i, j, _ in self.matrix) + 1
        max_col = max(j for i, j, _ in self.matrix) + 1
        matrix = [[0] * max_col for _ in range(max_row)]
        for i, j, value in self.matrix:
            matrix[i][j] = value
        return "\n".join(" ".join(map(str, row)) for row in matrix)


if __name__ == "__main__":
    # 示例矩阵
    matrix1 = np.random.rand(3, 3)
    matrix2 = np.random.rand(3, 3)

    # 创建稀疏矩阵对象
    sparse_matrix1 = SparseMatrix(matrix1)
    sparse_matrix2 = SparseMatrix(matrix2)

    # 执行加法
    result_matrix_add = sparse_matrix1.add(sparse_matrix2)

    # 执行乘法
    result_matrix_mul = sparse_matrix1.multiply(sparse_matrix2)

    # 打印结果
    print("Matrix 1:")
    print(sparse_matrix1)
    print("Matrix 2:")
    print(sparse_matrix2)
    print("Result Matrix:")
    print(result_matrix_add)
    print("Result Matrix:")
    print(result_matrix_mul)