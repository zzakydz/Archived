import random
import numpy as np
import matplotlib.pyplot as plt

# Inisialisasi variabel
a0 = 0.8
b0 = 3

# Variabel x model
x_model = np.array([2, 3, 5, 7, 11])

# Wadah y model
y_model = []

# Menghitung y berdasarkan persamaan y = ax + b
for x in x_model:
    y = a0 * x + b0
    y_model.append(y)

# Variabel baru untuk menyimpan hasil y_model + nilai random
y_model_adjusted = np.array([y + random.random() for y in y_model])

# Buat matriks d (data y)
d = y_model_adjusted.reshape(-1, 1)

# Buat matriks G (termasuk kolom x dan konstanta 1 untuk regresi linier)
G = np.column_stack((x_model, np.ones_like(x_model)))

# Hitung G transpose
Gt = G.T

# Hitung G transpose dikali G
GtG = Gt @ G

# Inverskan (GtG)^-1
GtG_inv = np.linalg.inv(GtG)

# Hitung model parameter (m) dengan least-squares
m = GtG_inv @ Gt @ d

# m berisi [a; b], di mana a adalah kemiringan dan b adalah intercept
a, b = m.flatten()

# Tampilkan hasil
print(f'Persamaan regresi: y = {a:.4f}x + {b:.4f}')
difference = y_model_adjusted - np.array(y_model)
print(f'data noise = {difference}')

# Plot data dan garis regresi
plt.scatter(x_model, y_model_adjusted, color='blue', label='With Noise', zorder=2)
plt.plot(x_model, a*x_model + b, color='red', linewidth=2, label='Garis Regresi', zorder=3)

# Plot model awal
plt.scatter(x_model, y_model, color='red', label='Without Noise', zorder=2)
plt.plot(x_model, a0*x_model + b0, color='black', linestyle='dashed', linewidth=2, label='Model Awal', zorder=1)

# Styling plot
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regresi Linear dengan Least-Squares')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
