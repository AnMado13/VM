import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math

def calculateWidder(image, variableImage, variableOriginal, order):
    derivativeImage = sp.diff(image, variableImage, order)
    multiplier = ((-1) ** order) * (variableImage ** (order + 1)) / math.factorial(order)
    widder = (multiplier * derivativeImage).subs(variableImage, order / variableOriginal)
    return widder

def calculateCoefficients(d, k):
    coefficients = np.ones(k)
    for j in range(0, k):
        for i in range(0, k):
            if (i == j):
                continue
            coefficients[j] *= d[j] / (d[j] - d[i])
    return coefficients

def calculateWidderBoosted(image, variableImage, variableOriginal, order, orderBoost):
    d = [j for j in range(1, orderBoost + 1)]
    c = calculateCoefficients(d, orderBoost)
    widderBoosted = 0
    for j in range(0, orderBoost):
        widderBoosted += c[j] * calculateWidder(image, variableImage, variableOriginal, order * d[j])
    return widderBoosted

def series(variableOriginal, order):
    partialSum = 0
    for k in range(0, order // 2):
        partialSum += ((-1) ** k) * ((variableOriginal ** (2 * k + 0.5)) / math.gamma(2 * k + 1.5))
    return partialSum
        
        
    

t = sp.symbols('p')
p = sp.symbols('p')
F = (sp.sqrt(p)) / (p ** 2 + 1)

#строим сетку, на которой будем строить график
N = 20
T = [0.1* i / N for i in range(0, N + 1)]

#разложение через ряд
n1 = 10
partialSum = series(t, n1)
solution_1 = [float(partialSum.subs(t, T[i])) for i in range(0, N + 1)]

#метод Виддера без ускорения сходимости
n2 = 10
widder = calculateWidder(F, p, t, n2)
solution_2 = [float(widder.subs(t, T[i])) for i in range(0, N + 1)]

#метод Виддера с ускорением сходимости
k = 3
n3 = 3
widderBoosted = calculateWidderBoosted(F, p, t, n3, k)
solution_3 = [float(widderBoosted.subs(t, T[i])) for i in range(0, N + 1)]

figure, axis = plt.subplots(1, 3)

axis[0].plot(T, solution_1)
axis[0].set_title("Разложение до первых %d слагаемых" % n1)

axis[1].plot(T, solution_2)
axis[1].set_title("Виддер без ускорения, n = %d" % n2)

axis[2].plot(T, solution_3)
axis[2].set_title("Виддер с ускорением, n = %d, k = %d" % (n3, k))

plt.show()
                  



