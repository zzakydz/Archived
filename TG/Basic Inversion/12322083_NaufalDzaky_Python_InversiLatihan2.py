# Naufal Dzaky Fadholi
# 12322083

import numpy as np
import matplotlib.pyplot as plt

# Data dari tabel
x = np.array([2, 4, 6, 8, 10, 12])
y = np.array([3.9572, 4.4854, 5.8003, 6.1419, 7.4218, 8.9157])
# matriks d (data y)
d = y.reshape(-1, 1)
# matriks G (termasuk kolom x dan konstanta 1 untuk regresi linier)
G = np.column_stack((x, np.ones_like(x)))
# G transpose
Gt = G.T
# G transpose dikali G
GtG = Gt @ G
# (GtG)^-1
GtG_inv = np.linalg.inv(GtG)
# model parameter (m) dengan least-squares
m = GtG_inv @ Gt @ d
# a adalah kemiringan dan b adalah intercept
a, b = m.flatten()
# estimasi y ketika x = 7.5
x_new = 7.5
y_new = a * x_new + b
# hasil
print(f'Persamaan regresi: y = {a:.4f}x + {b:.4f}')
print(f'Estimasi y untuk x = {x_new} adalah {y_new:.4f}')
# Plot
plt.scatter(x, y, color='blue', label='Data', zorder=2)
plt.plot(x, a*x + b, color='red', linewidth=2, label='Garis Regresi', zorder=3)
plt.scatter(x_new, y_new, color='green', marker='*', s=100, label=f'Estimasi (x={x_new})', zorder=4)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regresi Linear dengan Least-Squares')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
