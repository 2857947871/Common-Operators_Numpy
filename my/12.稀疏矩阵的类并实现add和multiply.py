import numpy as np

class SparseMatrix:
    def __init__(self, matrix):
        
        # 稀疏化
        self.matrix = []
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if matrix[i][j] != 0:
                    self.matrix.append((i, j, matrix[i][j]))

    def __str__(self):
        max_row = max(i for i, j, value in self.matrix) + 1
        max_col = max(j for i, j, value in self.matrix) + 1
        matrix = [[0] * max_col for _ in range(max_row)]
        for i, j, value in self.matrix:
            matrix[i][j] = value
        return "\n".join(" ".join(map(str, row)) for row in matrix)

    # other_matrix: 另一个class SparseMatrix
    def add(self, other_matrix):
        result = {(i, j): value for i, j, value in self.matrix}
        
        for i, j, value in other_matrix.matrix:
            if (i, j) in result:
                result[(i, j)] += value
            else:
                result[(i, j)]  = value
        
        # 结果转换为list
        max_row = max(i for i, j in result) + 1
        max_col = max(j for i, j in result) + 1
        
        return self.dict2list(max_row, max_col, result)

    def mul(self, other_matrix):
        self_rows = max(i for i, j, value in self.matrix) + 1
        self_cols = max(j for i, j, value in self.matrix) + 1
        other_rows = max(i for i, j, value in other_matrix.matrix) + 1
        other_cols = max(j for i, j, value in other_matrix.matrix) + 1
        
        if (self_cols != other_rows):
            raise ValueError("The number of columns in the first matrix must be equal to the number of rows in the second matrix")

        # 行: i, k   列: j, l
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

    def dict2list(self, max_row, max_col, result):
        result_matrix = [[0] * max_col for _ in range(max_row)]
        
        for (i, j), value in result.items():
            result_matrix[i][j] = value
        return SparseMatrix(result_matrix)   
        
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
    result_matrix_mul = sparse_matrix1.mul(sparse_matrix2)

    # 打印结果
    print("Matrix 1:")
    print(sparse_matrix1)
    print("Matrix 2:")
    print(sparse_matrix2)
    print("Result Matrix:")
    print(result_matrix_add)
    print("Result Matrix:")
    print(result_matrix_mul)