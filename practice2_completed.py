import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pylab import rcParams
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.api as smt


def resolve_data_path(filename: str) -> Path:
    """
    Ищет файл в нескольких типичных местах:
    - рядом со скриптом
    - в /mnt/data (текущее окружение)
    - в data/
    - в data/underwork/5/
    """
    candidates = [
        Path(__file__).resolve().parent / filename,
        Path("/mnt/data") / filename,
        Path.cwd() / filename,
        Path.cwd() / "data" / filename,
        Path.cwd() / "data" / "underwork" / "5" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Не найден файл {filename}. Проверил: {candidates}")


# ----------------------------
# Загрузка данных
# ----------------------------
calm_path = resolve_data_path("calm_p.csv")
passengers_path = resolve_data_path("passengers.csv")

tsdf_c = pd.read_csv(calm_path).set_index("Time").sort_index()
passengers = pd.read_csv(passengers_path)
passengers["Month"] = pd.to_datetime(passengers["Month"])
df = passengers.set_index("Month").sort_index()

print("Первые строки ЭКГ:")
print(tsdf_c.head())
print("\nОписательная статистика ЭКГ:")
print(tsdf_c.describe())

print("\nПервые строки пассажиропотока:")
print(df.head())
print("\nОписательная статистика пассажиропотока:")
print(df.describe())


# ----------------------------
# Графическое представление
# ----------------------------
def plot_assignation(axp, data, xlabel, ylabel, title1):
    axp.plot(data)
    axp.set_xlabel(xlabel)
    axp.set_ylabel(ylabel)
    axp.set_title(title1)


fig, axs = plt.subplots(2, 1, figsize=(20, 15))
fig.suptitle("Обследуемый # 5")

plot_assignation(axs[0], tsdf_c["1"], "time", "mV", "Покой. Отведение")
axs[1].plot(df["Passengers"])
axs[1].set_title("Passengers")
plt.show()


# ----------------------------
# Декомпозиция пассажиропотока
# ----------------------------
rcParams["figure.figsize"] = (11, 9)

decompose = seasonal_decompose(passengers["Passengers"], period=10)
decompose.plot()
plt.show()

new_ps = decompose.trend * (decompose.seasonal + 1) * decompose.resid

fig, axs = plt.subplots(figsize=(20, 15))
plt.plot(new_ps)
plt.title("Пассажиропоток после композиции компонент")
plt.show()

passengers_r = passengers["Passengers"] - decompose.trend
passengers_r.plot(label="Без тренда")
passengers["Passengers"].plot(label="Исходный ряд")
plt.legend()
plt.title("Пассажиропоток: исходный и без тренда")
plt.show()


# ==========================================================
# ЗАДАНИЕ 1: Сделайте декомпозицию для ЭКГ
# ==========================================================
Fs = 1000  # частота дискретизации
signal = tsdf_c["2"].astype(float).values

# Поиск пиков
peaks, _ = find_peaks(signal, distance=500)

rr_intervals = np.diff(peaks) / Fs
bpm = 60 / np.mean(rr_intervals)
period = int(round((60 / bpm) * Fs))

print(f"\nЭКГ: найдено пиков = {len(peaks)}")
print(f"Оценка BPM = {bpm:.2f}")
print(f"Период для декомпозиции = {period}")

ecg_series = tsdf_c["2"].astype(float)

# Разложение ЭКГ на компоненты
ecg_decompose = seasonal_decompose(
    ecg_series,
    period=period,
    model="additive",
    extrapolate_trend="freq"
)
ecg_decompose.plot()
plt.suptitle("Декомпозиция ЭКГ", y=1.02)
plt.show()

# Разложение белого шума
white_noise = np.random.normal(0, 1, 500)
noise_decompose = seasonal_decompose(white_noise, period=10, model="additive")
noise_decompose.plot()
plt.suptitle("Декомпозиция белого шума", y=1.02)
plt.show()


# ----------------------------
# Стационарность: пассажиры
# ----------------------------
alpha = 0.05
name = "Пассажиры"
ts = passengers["Passengers"]

print(f"\nТест Дики-Фуллера ряда {name}:")
dftest = adfuller(ts, autolag="AIC")
dfoutput = pd.Series(
    dftest[0:4],
    index=["Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"]
)

for key, value in dftest[4].items():
    dfoutput[f"Critical Value ({key})"] = value

print(dfoutput)

if dfoutput["p-value"] < alpha:
    print(f"Значение p меньше {alpha * 100}%. Ряд стационарный.")
else:
    print(f"Значение p больше {alpha * 100}%. Ряд не стационарный.")


# ==========================================================
# ЗАДАНИЕ 2: Проверьте на стационарность временной ряд с ЭКГ
# ==========================================================
# Для длинного ЭКГ-ряда ограничиваем maxlag=30, чтобы тест выполнялся быстро.
name = "ЭКГ"
ts_ecg = tsdf_c["2"].dropna().astype(float)

print(f"\nТест Дики-Фуллера ряда {name}:")
dftest_ecg = adfuller(ts_ecg, maxlag=30, autolag="AIC")
dfoutput_ecg = pd.Series(
    dftest_ecg[0:4],
    index=["Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"]
)

for key, value in dftest_ecg[4].items():
    dfoutput_ecg[f"Critical Value ({key})"] = value

print(dfoutput_ecg)

if dfoutput_ecg["p-value"] < alpha:
    print(f"Значение p меньше {alpha * 100}%. Ряд ЭКГ стационарный.")
else:
    print(f"Значение p больше {alpha * 100}%. Ряд ЭКГ не стационарный.")


# ----------------------------
# Тренд: пассажиры
# ----------------------------
window = 30

rolling_mean = ts.rolling(window=window).mean()
rolling_std = ts.rolling(window=window).std()

plt.figure(figsize=(15, 5))
plt.title(ts.name)
plt.plot(ts[window:], label="Реальные значения", color="black")

plt.plot(rolling_mean, label="MA" + str(window), color="red")

lower_bound = rolling_mean - (1.96 * rolling_std)
upper_bound = rolling_mean + (1.96 * rolling_std)

plt.fill_between(
    x=ts.index,
    y1=lower_bound,
    y2=upper_bound,
    color="lightskyblue",
    alpha=0.4
)
plt.legend(loc="best")
plt.grid(True)
plt.show()


# ==========================================================
# ЗАДАНИЕ 3: Повторите код выше для ЭКГ
# ==========================================================
ts_ecg = tsdf_c["2"].astype(float)

rolling_mean_ecg = ts_ecg.rolling(window=window).mean()
rolling_std_ecg = ts_ecg.rolling(window=window).std()

plt.figure(figsize=(15, 5))
plt.title("ECG signal (lead 2)")
plt.plot(ts_ecg.iloc[window:], label="Реальные значения", color="black")

plt.plot(rolling_mean_ecg, label="MA" + str(window), color="red")

lower_bound_ecg = rolling_mean_ecg - (1.96 * rolling_std_ecg)
upper_bound_ecg = rolling_mean_ecg + (1.96 * rolling_std_ecg)

plt.fill_between(
    x=ts_ecg.index,
    y1=lower_bound_ecg,
    y2=upper_bound_ecg,
    color="lightskyblue",
    alpha=0.4
)
plt.legend(loc="best")
plt.grid(True)
plt.show()


# ----------------------------
# Автокорреляция и коррелограмма
# ----------------------------
ts = passengers["Passengers"]

fig = plt.figure(figsize=(12, 7))
smt.graphics.plot_acf(ts, lags=30, alpha=0.5)
smt.graphics.plot_pacf(ts, lags=30, alpha=0.5)
plt.show()

fig = plt.figure(figsize=(20, 9))
layout = (2, 2)
ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
acf_ax = plt.subplot2grid(layout, (1, 0))
pacf_ax = plt.subplot2grid(layout, (1, 1))

ts.plot(ax=ts_ax)
ts_ax.set_title("Time Series Analysis Plots")
smt.graphics.plot_acf(ts, lags=30, ax=acf_ax, alpha=0.5)
smt.graphics.plot_pacf(ts, lags=30, ax=pacf_ax, alpha=0.5)

plt.tight_layout()
plt.show()
