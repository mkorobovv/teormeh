import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Глобальные параметры
m1 = 2  # масса стержня OC
m2 = 3  # масса диска
m3 = 1  # масса точки A
l = 1   # длина стержня OC
r = 0.5 # радиус диска
c = 0  # жесткость пружины
g = 9.8 # ускорение свободного падения

# Определяем систему ОДУ
def f(y, t):
    phi, psi, phit, psit = y
    
    a11 = (m1 / 3 + m2 + m3) * l
    a12 = m3 * r * np.cos(phi + psi)
    a1 = -(m1 / 2 + m2 + m3) * g * np.sin(phi) - c / l * (phi + psi) + m3 * r * psit**2 * np.sin(phi + psi)
    
    a21 = m3 * l * np.cos(phi + psi)
    a22 = (m2 / 2 + m3) * r
    a2 = m3 * g * np.sin(psi) - c / r * (phi + psi) + m3 * l * phit**2 * np.sin(phi + psi)
    
    A = np.array([
        [a11, a12], 
        [a21, a22]
        ])
    b = np.array([a1, a2])
    
    phitt, psitt = np.linalg.solve(A, b)
    
    return [phit, psit, phitt, psitt]

# Начальные условия и временные настройки
y0 = [-1, 1, 0.5, -0.5]
tstep = 0.01
tfin = 10
tout = np.arange(0, tfin + tstep, tstep)

# Решение системы ОДУ
sol = odeint(f, y0, tout)

phi = sol[:, 0]
psi = sol[:, 1]
phit = sol[:, 2]
psit = sol[:, 3]

# Расчет Rx и Ry
phitt = np.zeros_like(phi)
psitt = np.zeros_like(psi)

for i in range(len(tout)):
    res = f(sol[i, :], tout[i])
    phitt[i] = res[2]
    psitt[i] = res[3]

Rx = -(m1/2 + m2 + m3) * l * (phitt * np.sin(phi) + phit**2 * np.cos(phi)) + m3 * r * (psitt * np.sin(psi) + psit**2 * np.cos(psi)) - (m1 + m2 + m3) * g
Ry = (m1/2 + m2 + m3) * l * (phitt * np.cos(phi) - phit**2 * np.sin(phi)) + m3 * r * (psitt * np.cos(psi) - psit**2 * np.sin(psi))

# Построение графиков
fig, axs = plt.subplots(4, 1, figsize=(8, 12))
axs[0].plot(tout, phi)
axs[0].set_title('phi(t)')
axs[0].grid(True)

axs[1].plot(tout, psi)
axs[1].set_title('psi(t)')
axs[1].grid(True)

axs[2].plot(tout, Rx)
axs[2].set_title('Rx(t)')
axs[2].grid(True)

axs[3].plot(tout, Ry)
axs[3].set_title('Ry(t)')
axs[3].grid(True)

plt.tight_layout()

# Анимация
fig, ax = plt.subplots()
ax.set_aspect('equal', 'box')
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 1])

ax.plot([-3, 3], [0, 0], color='black')  # Потолок
ax.plot(0, 0, 'o', color='black', markerfacecolor='green')  # Шарнир O

line, = ax.plot([0, 5 * l * np.sin(phi[0])], [0, -5 * l * np.cos(phi[0])], color='black', linewidth=2)  # Стержень
D, = ax.plot([-5 * r * np.cos(psi[0]), 5 * r * np.cos(psi[0])] + 5 * l * np.sin(phi[0]), 
             [5 * r * np.sin(psi[0]), -5 * r * np.sin(psi[0])] - 5 * l * np.cos(phi[0]), color='blue')  # Диаметр диска

th = np.linspace(0, 2 * np.pi, 100)
x = 5 * l * np.sin(phi[0]) + 5 * r * np.cos(th)
y = -5 * l * np.cos(phi[0]) + 5 * r * np.sin(th)
h, = ax.plot(x, y, color='black')  # Диск

pointA, = ax.plot(5 * l * np.sin(phi[0]) + 5 * r * np.sin(phi[0]), 
                  -5 * l * np.cos(phi[0]) + 5 * r * np.cos(phi[0]), 'o', color='black', markerfacecolor='green')  # Точка A

alpha = np.arange(0, 6 * np.pi + phi[0], 0.01)
RPr = 0.01 + 0.5 * alpha / (6 * np.pi + phi[0])
XPr = 5 * l * np.sin(phi[0]) - RPr * np.sin(alpha)
YPr = -5 * l * np.cos(phi[0]) + RPr * np.cos(alpha)
Pruzhina, = ax.plot(XPr, YPr, color=[0.5, 0, 0])  # Спиральная пружина

def update(frame):
    i = frame
    line.set_data([0, 5 * l * np.sin(phi[i])], [0, -5 * l * np.cos(phi[i])])
    
    D.set_data([-5 * r * np.cos(psi[i]), 5 * r * np.cos(psi[i])] + 5 * l * np.sin(phi[i]), 
               [5 * r * np.sin(psi[i]), -5 * r * np.sin(psi[i])] - 5 * l * np.cos(phi[i]))
    
    x = 5 * l * np.sin(phi[i]) + 5 * r * np.cos(th)
    y = -5 * l * np.cos(phi[i]) + 5 * r * np.sin(th)
    h.set_data(x, y)
    
    pointA.set_data(5 * l * np.sin(phi[i]) + 5 * r * np.sin(psi[i]), 
                    -5 * l * np.cos(phi[i]) + 5 * r * np.cos(psi[i]))
    
    alpha = np.arange(0, 6 * np.pi + phi[i], 0.01)
    RPr = 0.01 + 0.5 * alpha / (6 * np.pi + phi[i])
    XPr = 5 * l * np.sin(phi[i]) - RPr * np.sin(alpha)
    YPr = -5 * l * np.cos(phi[i]) + RPr * np.cos(alpha)
    Pruzhina.set_data(XPr, YPr)
    return line, D, h, pointA, Pruzhina

ani = FuncAnimation(fig, update, frames=len(tout), interval=10, blit=True)
plt.show()
