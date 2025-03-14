% Naufal Dzaky Fadholi
% 12322083

% Data dari tabel
x = [2 4 6 8 10 12]';
y = [3.9572 4.4854 5.8003 6.1419 7.4218 8.9157]';
% 1. Buat matriks d (data y)
d = y;
% 2. Buat matriks G (termasuk kolom x dan konstanta 1 untuk regresi linier)
G = [x ones(size(x))];
% 3. Hitung G transpose
Gt = G';
% 4. Hitung G transpose dikali G
GtG = Gt * G;
% 5. Inverskan (GtG)^-1
GtG_inv = inv(GtG);
% 6. Hitung model parameter (m) dengan least-squares
m = GtG_inv * Gt * d;
% m berisi [a; b], di mana a adalah kemiringan dan b adalah intercept
a = m(1);
b = m(2);
% 7. Gunakan model untuk estimasi y ketika x = 7.5
x_new = 7.5;
y_new = a * x_new + b;
% Tampilkan hasil
fprintf('Persamaan regresi: y = %.4fx + %.4f\n', a, b);
fprintf('Estimasi y untuk x = %.1f adalah %.4f\n', x_new, y_new);
% 8. Plot data dan garis regresi
scatter(x, y, 'bo', 'filled'); % Plot titik data
hold on;
x_fit = linspace(min(x), max(x), 100);
y_fit = a * x_fit + b;
plot(x_fit, y_fit, 'r-', 'LineWidth', 2); % Plot garis regresi
plot(x_new, y_new, 'gx', 'MarkerSize', 10, 'LineWidth', 2); % Plot estimasi
xlabel('x');
ylabel('y');
title('Regresi Linear dengan Least-Squares');
legend('Data', 'Garis Regresi', 'Estimasi x=7.5', 'Location', 'Best');
grid on;
hold off;
