import sympy as sp
import prettytable as pt
import numpy as np

X = sp.symbols('x')
T = sp.symbols('t')
a = 1 #коэф. оператора
b = 1
alpha_1 = 1     #граничные коэф.
beta_2 = 1
U = X + T      #по конкретному решению строим задачу
L = a * sp.diff(U, X, X) + b * sp.diff(U, X)

f = sp.diff(U, T) - L #все эти функции заданы и известны
phi = U.subs(T, 0)
alpha = U.subs(X, 0)
beta = (sp.diff(U, X)).subs(X, 1)

def grid(N, M):
    h = 1 / N #создаем сетку переменных
    x = np.array([i * h for i in range(0, N + 1)])
 
    tau = 0.1 / M #создаем временную сетку
    t = np.array([k * tau for k in range(0, M + 1)])
    
    u = np.zeros((M + 1, N + 1))
    for i in range(0, N + 1):
        u[0, i] =  phi.subs(X, x[i]) #находим значения решения при t = 0
    for k in range(1, M + 1):   #находим значения решения при t = k
        for i in range(1, N):   #считаем некрайние значение разностной схемой
            Lh = a * (u[k - 1, i + 1] - 2 * u[k - 1, i] + u[k - 1, i - 1]) / (h ** 2) + b * (u[k - 1, i + 1] - u[k - 1, i - 1]) / (2 * h)
            u[k, i] = u[k - 1, i] + tau * (f.subs([(X, x[i]), (T, t[k-1])]) + Lh)
        u[k, 0]  = alpha.subs(T, t[k]) / alpha_1    #досчитываем крайние значения
        u[k, N] = (2 * h * beta.subs(T, t[k]) / (beta_2) + 4 * u[k, N - 1] - u[k, N - 2]) / 3
    return u


N, M = 5, 5     #берем кратные 5ки всегда для удобства
u = grid(N, M)

print("Крупная сетка:")
table = pt.PrettyTable()
table.add_column("x/t", [round(k * 0.1 / M, 2) for k in range(0, M + 1, M // 5)])
for i in range(0, N + 1, N // 5):
    table.add_column(str(round(i / N, 2)), u[:, i])
print(table)

print("Точность решения и внутренняя сходимость:")
table = pt.PrettyTable()
table.field_names = ["h", "tau", "||u_ex - u1||", "||u1 - u2||"]
for i in range(0, 3):
    u1 = grid(N, M) #решения
    u2 = grid(2 * N, 3 * M)
    dev1 = 0 #здесь будут погрешности
    dev2 = 0
    for k in range(0, M + 1, M // 5): #вычисляем погрешности на крупной сетке
        for i in range(0, N + 1, N // 5):
            if abs(u1[k, i] - U.subs([(X, i / N), (T, k * 0.1 / M)])) > dev1 :
                dev1 = abs(u1[k, i] - U.subs([(X, i / N), (T, k * 0.1 / M)]))
            if abs(u1[k, i] - u2[k * 3, i * 2]) > dev2:
                dev2 = abs(u1[k, i] - u2[k * 3, i * 2])
    table.add_row([1 / N, 0.1 / M, dev1, dev2])
    N *= 2
    M *= 3
    
print(table)
