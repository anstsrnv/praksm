import matplotlib.pyplot as plt
import numpy as np
import easygui
import math
from scipy.fftpack import fft, ifft


# Фильтр скользящего среднего (КИХ)
def moving_window1(s, wid):
    a = []
    s = [0]*wid+s
    for i in range(wid, len(s)):
        a.append(sum(s[i - wid:i]) / wid)
    return a


# Фильтр скользящего среднего (БИХ)
def moving_window2(s, wid):
    s = [0]*wid+s
    a = [0]*wid
    for i in range(wid, len(s)):
        a.append(a[i-1]+(s[i]-s[i-wid])/wid)
    return a


# Временной ряд дисперсий
def t_var(s, n):
    d = []
    m = np.mean(s)
    for el in s:
        d.append((el-m)**2)
    return d


# АКФ стандартным методом (кольцевого смещения)
def acf_cd(s, n):
    men = sum(s) / n
    signal1 = []
    r = []
    for i in range(n):
        signal1.append(((s[i] - men) ** 2) / (n - 1))
    dis = sum(signal1)
    k = 1 / (n * dis)
    for i in range(-n + 1, n):
        h = 0
        for j in range(0, n - abs(i)):
            h += k * (s[j + abs(i)] * s[j])
        r.append(h)
    return r


# АКФ при помощи БПФ
def acf_fft(s, n):
    k = math.log(n, 2)
    k = math.ceil(k)
    k = 2 ** k
    # Нахождение квадрата
    x2_a = ifft(fft(s, k) * np.conj(fft(s, k)))
    mann = fft(s, k) * np.conj(fft(s, k))
    m = max(x2_a)
    return x2_a / m, k


# Открытие файла с сигналом
file_name = easygui.fileopenbox(filetypes=["*.txt"])
with open(file_name) as file:
    signal = [float(row.strip().split()[2]) for row in file]

# Расчет интервала дискретизации
T_s = 1 / int(input('Введите частоту дискретизации (Гц): '))
n = int(input('Введите ширину окна: '))

# Названия графиков
titles_1 = ['Исходный сигнал', 'Сигнал после фильтрации скользящим окном (КИХ) шириной n = ' + str(n), 'Сигнал после фильтрации скользящим окном (БИХ) шириной n = ' + str(n)]
titles_2 = ['Исходный сигнал', 'Временной ряд дисперсий']
titles_3 = ['Исходный сигнал', 'АКФ кольцевым смещением', 'АКФ при помощи БПФ']


# Вывод графиков фильтр скользящего среднего
plt.figure(figsize=(24, 8))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(titles_1[i], fontsize=9)
    if i == 0:
        y = signal
        interval = [i * T_s for i in range(len(signal))]
    if i == 1:
        y = moving_window1(signal, n)
        #interval = interval[:-n]
    if i == 2:
        y = moving_window2(signal, n)
        y = y[n:]
        #interval = [i * T_s for i in range(len(signal))]
    plt.plot(interval, y)

# Вывод графика временного ряда дисперсий
plt.figure(figsize=(24, 8))
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(titles_2[i])
    interval = [i * T_s for i in range(len(signal))]
    if i == 0:
        y = signal
    if i == 1:
        y = t_var(signal, len(signal))
    plt.plot(interval, y)

# Вывод графиков АКФ
plt.figure(figsize=(24, 8))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(titles_3[i])
    if i == 0:
        y = signal
        interval = [i * T_s for i in range(len(signal))]
    if i == 1:
        y = acf_cd(signal, len(signal))
        interval = [i * T_s for i in range(-len(signal), len(signal) - 1)]
    if i == 2:
        y, inter2 = acf_fft(signal, len(signal))
        interval = [i * 2 * T_s for i in range(-int(inter2 / 2), int(inter2 / 2))]
    plt.plot(interval, y)
plt.show()
