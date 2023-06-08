import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def equation(y, t):
    y0, y1 = y
    dydt = [y1, -2*y0 - 3*y1]  # 这里是二阶微分方程的形式
    return dydt

# 初始条件
y0 = [0, 1]
# 时间点
t = np.linspace(0, 10, 100)

# 解微分方程
sol = odeint(equation, y0, t)

# 提取解
y0_sol = sol[:, 0]
y1_sol = sol[:, 1]

# 绘制结果
plt.plot(t, y0_sol, 'b', label='y0(t)')
plt.plot(t, y1_sol, 'r', label='y1(t)')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.show()
