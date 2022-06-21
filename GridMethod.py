import sympy as sp
import prettytable as pt
import numpy as np

X = sp.symbols('x')
T = sp.symbols('t')
a = 1 #коэф. оператора
b = 1
alpha_1 = 1     #граничные коэф.
beta_2 = 1
U = X ** 2 + T ** 2      #по конкретному решению строим задачу
L = a * sp.diff(U, X, X) + b * sp.diff(U, X)

f = sp.diff(U, T) - L #все эти функции заданы и известны
phi = U.subs(T, 0)
alpha = U.subs(X, 0)
beta = (sp.diff(U, X)).subs(X, 1)

def run_through(A, G, N):
    s = np.zeros(N)
    t = np.zeros(N)
    y = np.zeros(N + 1)
    s[0] = A[0, 1] / A[0, 0]
    t[0] = - G[0] / A[0, 0]
    for i in range(1, N):
        s[i] =  A[i, i + 1] / (A[i, i] - A[i, i - 1] * s[i - 1])
        t[i] = (A[i, i - 1] * t[i - 1] - G[i]) / (A[i, i] - A[i, i - 1] * s[i - 1])

    y[N] = (A[N, N - 1] * t[N - 1] - G[N]) / (A[N, N] - A[N, N - 1] * s[N - 1])
    for i in range(N - 1, -1, -1):
        y[i] = s[i] * y [i + 1] + t[i]
    return y
        
def grid_e(M, N):
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

def grid_i(sigma, M, N):
    h = 1 / N #создаем сетки
    x = np.array([i * h for i in range(0, N + 1)])
    tau = 0.1 / M
    t = np.array([k * tau for k in range(0, M + 1)])

    u = np.zeros((M + 1, N + 1))
    for i in range(0, N + 1):
        u[0, i] =  phi.subs(X, x[i]) #находим значения решения при t = 0
    for k in range(1, M + 1):
        A = np.zeros((N + 1, N + 1)) #Будущая трехдиагональная матрица системы
        G = np.zeros(N + 1) #Правая часть системы
        G[0] = alpha.subs(T, t[k])
        G[N] = beta.subs(T, t[k])   #Заполняем матрицу системы
        A[0, 0] = - 1
        A[N, N - 1] = - 1 / h
        A[N, N] = - 1 / h
        for i in range(1, N):
            Lh = a * (u[k - 1, i + 1] - 2 * u[k - 1, i] + u[k - 1, i - 1]) / (h ** 2) + b * (u[k - 1, i + 1] - u[k - 1, i - 1]) / (2 * h)
            G[i] = - 1 / tau * u[k - 1, i] - (1 - sigma) * Lh - f.subs([(X, x[i]), (T, t[k])])
            A[i, i - 1] = sigma * a / (h ** 2) - sigma * b / (2 * h)
            A[i, i] = 1 / tau + 2 * sigma * a / (h ** 2)
            A[i, i + 1] = sigma * a / (h ** 2) + sigma * b / (2 * h)
        u[k, :] = run_through(A, G, N) #Решаем систему методом прогонки
    return u

def Big_Table(u, M, N):
    print("1)Крупная сетка:")
    table = pt.PrettyTable()
    table.add_column("x/t", [round(k * 0.1 / M, 2) for k in range(0, M + 1, M // 5)])
    for i in range(0, N + 1, N // 5):
        table.add_column(str(round(i / N, 2)), u[:, i])
    print(table)

def AandS_Table(sigma, u, M, N):
    print("2)Точность решения и внутренняя сходимость:")
    table = pt.PrettyTable()
    table.field_names = ["h", "tau", "||u_ex - u1||", "||u1 - u2||"]
    m = 3
    n = 2
    for i in range(0, 3):
        if sigma == 0:  #находим решения каким-то из методов
            u1 = grid_e(M, N) 
            u2 = grid_e(m * M, n * N)
        else:
            u1 = grid_i(sigma, M, N)
            u2 = grid_i(sigma, m * M, n * N)
        dev1 = 0 #здесь будут погрешности
        dev2 = 0
        for k in range(0, M + 1, M // 5): #вычисляем погрешности на крупной сетке
            for i in range(0, N + 1, N // 5):
                if abs(u1[k, i] - U.subs([(X, i / N), (T, k * 0.1 / M)])) > dev1 :
                    dev1 = abs(u1[k, i] - U.subs([(X, i / N), (T, k * 0.1 / M)]))
                if abs(u1[k, i] - u2[k * m, i * n]) > dev2:
                    dev2 = abs(u1[k, i] - u2[k * m, i * n])
        table.add_row([1 / N, 0.1 / M, dev1, dev2])
        M *= m
        N *= n
    print(table)

M, N = 5, 5     #берем кратные 5ки всегда для удобства

print("Явная схема:")
u = grid_e(M, N)
Big_Table(u, M, N)
AandS_Table(0, u, M, N)

print("\nНеявная схема sigma = 1:")
u = grid_i(1, M, N)
Big_Table(u, M, N)
AandS_Table(1, u, M, N)

print("\nНеявная схема sigma = 1/2:")
u = grid_i(0.5, M, N)
Big_Table(u, M, N)
AandS_Table(0.5, u, M, N)



