import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def standartRegularization(matrix, rightPart, size, parameter):
    matrixTranspose = matrix.transpose()
    regMatrix = (matrixTranspose).dot(matrix) + parameter * np.eye(size)
    regRightPart = matrixTranspose.dot(rightPart)
    
    return np.linalg.solve(regMatrix, regRightPart)

def standartRegularizationWithAccuracy(matrix, rightPart, size, accuracy):
    solution = np.zeros(size)
    delta = accuracy
    error = accuracy + 1
    while (error > accuracy):
        solution = standartRegularization(matrix, rightPart, size, delta)
        error = np.linalg.norm(matrix.dot(solution) - rightPart, np.inf)
        delta /= 5
    return solution

x = sp.symbols('x')
s = sp.symbols('s')

Ker = sp.cos(3 * s * x)
z = s * (1 - s) #или s * (1 - s)
g = Ker * z #подинтегральное выражение

u = sp.integrate(g, (s, 0, 1)) #правая часть

n = 15
S = np.array([(k + 0.5) / n for k in range(0, n)]) #будем использовать формулу средних прямоугольников
solutionEx = np.array([float(z.subs(s, S[j])) for j in range(0, n)]) #точное решение в точках
A_k = 1 / n #так как формула средних прямоугольников

U = np.array([float(u.subs(x, S[j])) for j in range(0, n)]) #правая часть СЛАУ
C = np.zeros((n, n)) #матрица СЛАУ
for j in range(0, n):
    KerLayer = Ker.subs(x, S[j])
    for k in range(0, n):
        C[j , k] = A_k * float(KerLayer.subs(s, S[k]))

epsilon = 0.00001
solutionReg = standartRegularizationWithAccuracy(C, U, n, epsilon)

fig, ax = plt.subplots()
ax.plot(S, solutionReg, linewidth=2.0)
ax.set(xlim=(0, 1.25), ylim=(0, 1.5))
plt.show()


