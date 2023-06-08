from sympy import symbols, Function, pi, dsolve, Eq, Derivative

# 定义未知函数和自变量
x = symbols('x')
y = Function('y')(x)

# 定义常数
c1 = symbols('c1')

# 定义微分方程
diff_eq = Eq(Derivative(y, x, x), 80 * pi * (pi/9) * 0.95 * 0.95 * y - 4 * pi * pi * (y - 0.113556))

# 定义初始条件
initial_conditions = {y.subs(x, 0): 0, y.diff(x).subs(x, 0): 2 * pi * (0.95 * (400/270) - 1)}

# 求解微分方程的解析解
solutions = dsolve(diff_eq, y, ics=initial_conditions)

# 输出解析解
for solution in solutions:
    print(solution.rhs + c1)