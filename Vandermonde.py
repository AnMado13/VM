import numpy as np
import prettytable as pt

def vandermondeMatrix(size):
    matrix = np.zeros((size, size))
    for k in range(1, n + 1):
        a_k = k
        for j in range(1, n + 1):
            b_j = j / 3
            matrix[k - 1, j - 1] = a_k ** b_j
            
    return matrix

def isOscillatory(matrix, size):
    for i in range(1, size + 1):
        if np.linalg.det(matrix[0:i, 0:i]) < 0:
            return False
        
    return True

def sqrtMatrix(matrix, size):
    lambda_1 = np.zeros((size, size))
    eigValues, eigVectors = np.linalg.eig(matrix)
    for i in range(0, n):
        lambda_1[i, i] = np.sqrt(eigValues[i])
        
    return eigVectors.dot(lambda_1).dot(np.linalg.inv(eigVectors))

def standartRegularization(matrix, parameter, rightPart, size):
    matrixTranspose = matrix.transpose()
    regMatrix = (matrixTranspose).dot(matrix) + parameter * np.eye(size)
    regRightPart = matrixTranspose.dot(rightPart)
    
    return np.linalg.solve(regMatrix, regRightPart)

def sqrtRegularization(sqrtMatrix, parameter, rightPart, size):
    sqrtMatrixTranspose = sqrtMatrix.transpose()
    regMatrix = (sqrtMatrixTranspose).dot(sqrtMatrix) + parameter * np.eye(size)
    regRightPart = sqrtMatrixTranspose.dot(np.linalg.inv(sqrtMatrix)).dot(rightPart)

    return np.linalg.solve(regMatrix, regRightPart)



n = 2
n_max = 10
epsilon = 0.0001 #допустимая погрешность регуляризации
normDifference = 0

table = pt.PrettyTable()
table.field_names = ["n", "Osc. type?", "cond(A)", "cond(B)", "||A - B*B||", "Кол-во шагов для Standart", "Кол-во шагов для Sqrt"]

while (n <= n_max):
    A = vandermondeMatrix(n)
    B = sqrtMatrix(A, n)

    z_0 = np.array([1 for i in range(0, n)])
    u = A.dot(z_0) #правая часть уравнения
    
    delta = 0.1
    standartCount = 0
    standartError = 1
    while (standartError > epsilon):
        standartRegSolution = standartRegularization(A, delta, u, n)
        standartError = np.linalg.norm(A.dot(standartRegSolution) - u, np.inf)
        delta /= 10
        standartCount += 1
        
    delta = 0.1
    sqrtCount = 0
    sqrtError = 1
    while (sqrtError > epsilon):
        sqrtRegSolution = sqrtRegularization(B, delta, u, n)
        sqrtError = np.linalg.norm(A.dot(sqrtRegSolution) - u, np.inf)
        delta /= 10
        sqrtCount += 1

    normDifference = np.linalg.norm(A - B.dot(B), np.inf)
    condA = np.linalg.norm(A, np.inf) * np.linalg.norm(np.linalg.inv(A), np.inf)
    condB = np.linalg.norm(B, np.inf) * np.linalg.norm(np.linalg.inv(B), np.inf)
    table.add_row([n, isOscillatory(A, n), condA, condB, normDifference, standartCount, sqrtCount])

    n += 1

print(table)

