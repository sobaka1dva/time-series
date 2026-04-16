# Импортируем необходимые библиотеки

import os
from os import path
import warnings
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.tsa.api as smt

"""Рутина с датасетами"""

# загрузите данные о пассажирах
# Ищем файл passengers.csv в нескольких типовых местах
candidate_paths = [
    'passengers.csv',
    './passengers.csv',
    '/mnt/data/passengers.csv',
    'data/passengers.csv',
    '/data/passengers.csv'
]
passengers_path = next((p for p in candidate_paths if os.path.exists(p)), None)
if passengers_path is None:
    raise FileNotFoundError('Не найден файл passengers.csv')

passengers = pd.read_csv(passengers_path)
# неподходящий формат данных приводим к тому, с которым Pandas может работать
passengers['Month'] = pd.to_datetime(passengers['Month'])
# также устанавливаем индекс и сортируем
df = passengers.set_index('Month').sort_index()

df.describe()

"""Отрисовываем временной ряд"""

plt.plot(df["Passengers"])

"""## Стационарный процесс  <a class="anchor" id="stacionary"></a>

Стационарный процесс - это случайный процесс, безусловное совместное распределение вероятностей которого не изменяется при сдвиге во времени. Следовательно, такие параметры, как среднее значение и дисперсия, также не меняются со временем, поэтому стационарные временные ряды легче прогнозировать.

Есть несколько способов установить, является ли временной ряд стационарным или нет, наиболее распространенными являются старая добрая визуализация, просмотр автокорреляции и выполнение статистических тестов.

Наиболее распространенным тестом является тест Дики-Фуллера (также называемый тест ADF), где нулевая гипотеза состоит в том, что временной ряд имеет единичный корень, другими словами, временной ряд не является стационарным.

Мы проверим, можно ли отвергнуть нулевую гипотезу, сравнив значение p с выбранным порогом (α), чтобы, если значение p меньше, мы могли отклонить нулевую гипотезу и предположить, что временной ряд с уверенностью является стационарным. уровень 1-α (технически мы просто не можем сказать, что это не так)

Временной ряд имеет единичный корень, или порядок интеграции один, если его первые разности образуют стационарный ряд. Это условие записывается как
$y_t\thicksim I(1)$ если ряд первых разностей $\triangle y_t=y_t-y_{t-1}$ является стационарным $\triangle y_t\thicksim I(0)$.

При помощи этого теста проверяют значение коэффициента $a$ в  авторегрессионном уравнении первого порядка AR(1)
$y_t=a\\cdot y_{t-1}+\\varepsilon_t,$
где $y_t$ — временной ряд, а $\varepsilon$— ошибка.

Если $a=1$, то процесс имеет единичный корень, в этом случае ряд $y_t$ не стационарен, является интегрированным временным рядом первого порядка $I(1)$. Если $|a|<1$, то ряд стационарный $I(0)$.
"""

# импортируем функцию, описывающую тест Дики-Фуллера
from statsmodels.tsa.stattools import adfuller

# всю теорию, описанную выше, реализуем с помощью statsmodels для проверки
# временного ряда перевозок на стационарность

alpha = 0.05
name = "Пассажиры"

# определяем временной ряд отдельной переменной
ts = df["Passengers"]

print(f'Тест Дики-Фуллера ряда {name} :')
# определяем результат значения теста из библиотеки с учетом
dftest = adfuller(ts, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)

if dfoutput["p-value"] < alpha:
    print(f"Значение p меньше {alpha * 100}%. Ряд стационарный.")
else:
    print(f"Значение p больше {alpha*100}%. Ряд не стационарный.")

"""А теперь попробуем осуществить дифференцирование. Перед этим опять попробуем декомпозицию ряда."""

# импортируем функцию seasonal_decompose из statsmodels
# (то есть осуществляем декомпозицию сигнала/временного ряда)
from statsmodels.tsa.seasonal import seasonal_decompose

# задаем размер графика
from pylab import rcParams
rcParams['figure.figsize'] = 11, 9


# примените функцию seasonal_decompose к данным о перевозках
decompose = seasonal_decompose(passengers["Passengers"],
                               period=6)
decompose.plot()
plt.show()

"""Создадим два временных ряда на основе имеющегося, только без тренда и сезонности.

Удаляем тренд согласно формуле: $y' = y_t - y_{t-1}$;

Удаляем сезонность согласно формуле: $y' = y_t - y_{t-s}$;
"""

nottrend = []
s = 6
notseason = []

# выборка без тренда
for i in range(1, len(df["Passengers"])):
    nottrend.append(df["Passengers"].iloc[i] - df["Passengers"].iloc[i-1])

# выборка без сезонности
for i in range(s, len(df["Passengers"])):
    notseason.append(df["Passengers"].iloc[i] - df["Passengers"].iloc[i-s])

# отрисовываем временной ряд без тренда
plt.plot(nottrend)

"""Теперь проведем тест Дики-Фуллера на временном ряде без тренда"""

alpha = 0.05
name = "Пассажиры без тренда"

ts = nottrend

print(f'Тест Дики-Фуллера ряда {name} :')
dftest = adfuller(ts, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)

if dfoutput["p-value"] < alpha:
    print(f"Значение p меньше {alpha * 100}%. Ряд стационарный.")
else:
    print(f"Значение p больше {alpha*100}%. Ряд не стационарный.")

# отрисовываем временной ряд без сезонности
plt.plot(notseason)

"""Аналогичным образом проведем тест Дики-Фуллера на временном ряде без сезонности"""

alpha = 0.05
name = "Пассажиры без сезона"

ts = notseason

print(f'Тест Дики-Фуллера ряда {name} :')
dftest = adfuller(ts, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)

if dfoutput["p-value"] < alpha:
    print(f"Значение p меньше {alpha * 100}%. Ряд стационарный.")
else:
    print(f"Значение p больше {alpha*100}%. Ряд не стационарный.")

"""## Преобразование Бокса-Кокса <a class="anchor" id="boxcox"></a>"""

# Преобразование Бокса-Кокса
from scipy.stats import boxcox

# вызываем функцию преобразования, которая выдает преобразованные данные и
# лучший параметр лямбда, который обеспечивает близость к нормальному
# распределению
transformed_data, best_lambda = boxcox(df["Passengers"])

# а теперь посмотрим на преобразованные данные
plt.plot(transformed_data)

"""Попробуем теперь из преобразованного временного ряда удалить тренд и
визуализировать его
"""

pnottrend = []

for i in range(1, len(transformed_data)):
   pnottrend.append(transformed_data[i] - transformed_data[i-1])


plt.plot(pnottrend)

"""Удалив тренд из преобразованного Боксом-Коксом ряда, попробуем опять проверить его на стационарность. Что-то изменилось?"""

alpha = 0.05
name = "Пассажиры после Кокса-Бокса"

ts = pnottrend

print(f'Тест Дики-Фуллера ряда {name} :')
dftest = adfuller(ts, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)

if dfoutput["p-value"] < alpha:
    print(f"Значение p меньше {alpha * 100}%. Ряд стационарный.")
else:
    print(f"Значение p больше {alpha*100}%. Ряд не стационарный.")

"""## Модели для предсказания значений временного ряда <a class="anchor" id="predict"></a>

Первой моделью будет модель **AR**, или же autoregression - модель, которая использует связь между наблюдением и некоторым количеством предыдущих наблюдений.​

Сделаем случайный ряд и затем поработаем с ним.
"""

# AR(1)

N = 500

ar1 = [1]

for i in range(1, N):
    ar1.append(0.76 * ar1[i-1] + np.random.random())

plt.plot(ar1)

"""Посмотрим какие у него стандартное отклонение и среднее."""

print(f"standart deviation = {np.std(ar1)}\n mean = {np.mean(ar1)}")

"""А теперь обернем его в датафрейм и посмотрим его обычную и частичную автокорреляцию. Что можно сказать по поводу этого временного ряда, глядя на эти параметры?"""

ts = pd.DataFrame(ar1)

fig = plt.figure(figsize=(20, 9))
layout = (2, 2)
ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
acf_ax = plt.subplot2grid(layout, (1, 0))
pacf_ax = plt.subplot2grid(layout, (1, 1))

ts.plot(ax=ts_ax)
ts_ax.set_title('Time Series Analysis Plots')
smt.graphics.plot_acf(ts, lags=20, ax=acf_ax, alpha=0.5)
smt.graphics.plot_pacf(ts, lags=10, ax=pacf_ax, alpha=0.5)
None

"""Сделаем ещё один случайный ряд, но уже и с отрицательными значениями коэффициента"""

# AR(1)

N = 500

ar2 = [1]

for i in range(1, N):
    ar2.append(- 0.76*ar2[i-1] + np.random.random())

plt.plot(ar2)

print(f"standart deviation = {np.std(ar2)}\n mean = {np.mean(ar2)}")

"""А что можно сказать по поводу этого ряда?"""

ts = pd.DataFrame(ar2)

fig = plt.figure(figsize=(20, 9))
layout = (2, 2)
ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
acf_ax = plt.subplot2grid(layout, (1, 0))
pacf_ax = plt.subplot2grid(layout, (1, 1))

ts.plot(ax=ts_ax)
ts_ax.set_title('Time Series Analysis Plots')
smt.graphics.plot_acf(ts, lags=20, ax=acf_ax, alpha=0.5)
smt.graphics.plot_pacf(ts, lags=10, ax=pacf_ax, alpha=0.5)
None

"""А теперь коэффициент >1"""

# AR(1)

N = 500

ar3 = [1]

for i in range(1, N):
    ar3.append(2 * ar3[i-1] + np.random.random())

plt.plot(ar3)

print(f"standart deviation = {np.std(ar2)}\n mean = {np.mean(ar2)}")

"""Ладно, пора возвращаться к прогнозированию. Следаем прогноз с помощью AR модели, предварительно поделив выборки на обучающую, валидационную и тестовую."""

df = pd.read_csv(passengers_path, names=["n","x"], skiprows=1)


df['t'] = df.index.values

ln = len(df)

# указываем 'объемы' выборок
train_cutoff = int(round(ln*0.75, 0))
validate_cutoff = int(round(ln*0.90,0))

# делим выборки
train_df = df[df['t'] <= train_cutoff]
validate_df = df[(df['t'] > train_cutoff) & (df['t'] <= validate_cutoff)]
forecast_df = df[df['t'] > validate_cutoff]

"""Визуализируем поделенные выборки.

* Обучающая выборка - синим цветом
* Валидационная выборка - оранжевым цветом
* Предсказываемая выборка - зеленым цветом
"""

plt.plot(train_df.t, train_df.x, label='Training data')
plt.plot(validate_df.t, validate_df.x, label='Validate data')
plt.plot(forecast_df.t, forecast_df.x, label='For prediction')
plt.legend()
plt.title('Airline passengers by month')
plt.ylabel('Total passengers')
plt.xlabel('Month')
plt.show()

from statsmodels.tsa.ar_model import AutoReg, ar_select_order

# создаем объект модели на основе данных временного ряда с 3 лагами
mod = AutoReg(df.x, 3, old_names=False)
# обучаем
res = mod.fit()

# выводим сводку информации об авторегрессионной модели
print(res.summary())

# опять обучаем модель, но на этот раз указываем тип ковариационной оценки
res = mod.fit(cov_type="HC0")

# смотрим, что изменилось
print(res.summary())

"""Продолжаем экспериментировать"""

sel = ar_select_order(df.x, 13, old_names=False)
sel.ar_lags
res = sel.model.fit()
print(res.summary())

"""Смотрим, что он предсказал"""

fig = res.plot_predict(train_cutoff)

"""Формируем предсказанные временные ряды"""

pred = res.predict(start=0, end=train_cutoff, dynamic=False)
v_pred = res.predict(start=train_cutoff+1, end=(validate_cutoff), dynamic=False)
f_pred = res.predict(start=validate_cutoff + 1, end=forecast_df.index[-1], dynamic=False)

"""Отрисовываем их"""

plt.plot(train_df.t, train_df.x, label='Training data')
plt.plot(validate_df.t, validate_df.x, label='Validate data')
plt.plot(forecast_df.t, forecast_df.x, label='For prediction')
plt.plot(validate_df.t, v_pred, label='Validate prediction ')
plt.plot(forecast_df.t, f_pred, label='Forecast prediction')
plt.plot(train_df.t, pred, label='Train prediction')

plt.legend()
plt.title('Airline passengers by month')
plt.ylabel('Total passengers')
plt.xlabel('Month')
plt.show()

plt.plot(pred)

# MA

df['t'] = df.index.values

ln = len(df)

# указываем 'объемы' выборок
train_cutoff = int(round(ln*0.75, 0))
validate_cutoff = int(round(ln*0.90,0))

# делим выборки
train_df = df[df['t'] <= train_cutoff]
validate_df = df[(df['t'] > train_cutoff) & (df['t'] <= validate_cutoff)]
forecast_df = df[df['t'] > validate_cutoff]

plt.plot(train_df["t"], train_df["x"], label="original data")
plt.plot(train_df["t"], train_df["x"].rolling(10).mean(), label="MA")
plt.legend()
plt.ylabel('Total passengers')
plt.xlabel('Month')
plt.show()

"""## Метрики точности прогноза <a class="anchor" id="metrics"></a>

* R2- коэффициент детерминации     ​
* MSE (RMSE) – mean squared error – среднеквадратичная ошибка​
* MAE – mean absolute error – средняя абсолютная ошибка​
* MAPE – mean absolute percentage error – средняя абсолютная ошибка в %​
* SMAPE – symmetric mean absolute percentage error – симметричная средняя абсолютная ошибка в %

Определяем метрики точности прогноза из библиотеки sklearn. Попробуй определить последнюю оставшуюся метрику **SMAPE** самостоятельно.
"""

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score

"""Вычислим значения ошибок модели AR, опираясь на предсказанные ею значения forecast."""

print("RMSE:", np.sqrt(mean_squared_error(forecast_df.x, f_pred)))
print("MAPE:", mean_absolute_percentage_error(forecast_df.x, f_pred))
print("MAE:", mean_absolute_error(forecast_df.x, f_pred))
print("R2: ", r2_score(forecast_df.x, f_pred))

"""# **Задание:** Изучить, как работает модель авторегрессии (AR) на временном ряду и оценить её качество."""

# Простыми словами, нужно сделать задание по аналогии с тем нестационарным временным рядом (см. выше).

from statsmodels.tsa.arima_process import arma_generate_sample

np.random.seed(42)

# AR(2) процесс
ar_data = arma_generate_sample(
    ar=np.array([1.0, -0.5, 0.7]),
    ma=np.array([1]),
    nsample=200,
    scale=1,
    burnin=1000
)

time = np.arange(200)

trend = time * 0.2
seasonality = 2 * np.sin(2 * np.pi * time / 12)

time_series = trend + seasonality + ar_data

df = pd.DataFrame({
    "t": time,
    "x": time_series
})

plt.figure(figsize=(10,4))
plt.plot(df.t, df.x, label="Original")
plt.title("Synthetic time series (trend + seasonality + AR)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0
    return 100 * np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask])

print("\n=== ЗАДАНИЕ: исследование AR-модели на синтетическом ряде ===")

alpha = 0.05
name = "Синтетический ряд"

ts = df["x"]

print(f'Тест Дики-Фуллера ряда {name} :')
dftest = adfuller(ts, autolag='AIC')
dfoutput = pd.Series(
    dftest[0:4],
    index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
)
for key, value in dftest[4].items():
    dfoutput[f'Critical Value ({key})'] = value
print(dfoutput)

if dfoutput["p-value"] < alpha:
    print(f"Значение p меньше {alpha * 100}%. Ряд стационарный.")
else:
    print(f"Значение p больше {alpha * 100}%. Ряд не стационарный.")

# Декомпозиция
decompose = seasonal_decompose(df["x"], period=12, model="additive")
decompose.plot()
plt.show()

# Удаляем тренд и сезонность
nottrend = df["x"].diff().dropna()
s = 12
notseason = df["x"].diff(s).dropna()

plt.figure(figsize=(10, 4))
plt.plot(nottrend)
plt.title("Синтетический ряд без тренда")
plt.grid(alpha=0.3)
plt.show()

print('Тест Дики-Фуллера ряда Синтетический ряд без тренда :')
dftest = adfuller(nottrend, autolag='AIC')
dfoutput = pd.Series(
    dftest[0:4],
    index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
)
for key, value in dftest[4].items():
    dfoutput[f'Critical Value ({key})'] = value
print(dfoutput)
if dfoutput["p-value"] < alpha:
    print(f"Значение p меньше {alpha * 100}%. Ряд стационарный.")
else:
    print(f"Значение p больше {alpha * 100}%. Ряд не стационарный.")

plt.figure(figsize=(10, 4))
plt.plot(notseason)
plt.title("Синтетический ряд без сезонности")
plt.grid(alpha=0.3)
plt.show()

print('Тест Дики-Фуллера ряда Синтетический ряд без сезонности :')
dftest = adfuller(notseason, autolag='AIC')
dfoutput = pd.Series(
    dftest[0:4],
    index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
)
for key, value in dftest[4].items():
    dfoutput[f'Critical Value ({key})'] = value
print(dfoutput)
if dfoutput["p-value"] < alpha:
    print(f"Значение p меньше {alpha * 100}%. Ряд стационарный.")
else:
    print(f"Значение p больше {alpha * 100}%. Ряд не стационарный.")

# Box-Cox для ряда с возможными отрицательными значениями
shift_value = abs(df["x"].min()) + 1 if df["x"].min() <= 0 else 0
transformed_data, best_lambda = boxcox(df["x"] + shift_value)

plt.figure(figsize=(10, 4))
plt.plot(transformed_data)
plt.title("Box-Cox преобразованный синтетический ряд")
plt.grid(alpha=0.3)
plt.show()

pnottrend = pd.Series(transformed_data).diff().dropna()

plt.figure(figsize=(10, 4))
plt.plot(pnottrend)
plt.title("Box-Cox ряд без тренда")
plt.grid(alpha=0.3)
plt.show()

print('Тест Дики-Фуллера ряда Синтетический ряд после Бокса-Кокса и удаления тренда :')
dftest = adfuller(pnottrend, autolag='AIC')
dfoutput = pd.Series(
    dftest[0:4],
    index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
)
for key, value in dftest[4].items():
    dfoutput[f'Critical Value ({key})'] = value
print(dfoutput)
if dfoutput["p-value"] < alpha:
    print(f"Значение p меньше {alpha * 100}%. Ряд стационарный.")
else:
    print(f"Значение p больше {alpha * 100}%. Ряд не стационарный.")

print(f"Сдвиг для Box-Cox: {shift_value:.6f}")
print(f"Лучшая lambda Box-Cox: {best_lambda:.6f}")

# Делим на train / validate / forecast
df["t"] = df.index.values
ln = len(df)
train_cutoff = int(round(ln * 0.75, 0))
validate_cutoff = int(round(ln * 0.90, 0))

train_df = df[df["t"] <= train_cutoff].copy()
validate_df = df[(df["t"] > train_cutoff) & (df["t"] <= validate_cutoff)].copy()
forecast_df = df[df["t"] > validate_cutoff].copy()

plt.figure(figsize=(10, 4))
plt.plot(train_df.t, train_df.x, label='Training data')
plt.plot(validate_df.t, validate_df.x, label='Validate data')
plt.plot(forecast_df.t, forecast_df.x, label='For prediction')
plt.legend()
plt.title('Synthetic series split')
plt.ylabel('x')
plt.xlabel('t')
plt.grid(alpha=0.3)
plt.show()

# Обучаем AR-модель на train
lag_order = 12
mod = AutoReg(train_df["x"], lags=lag_order, old_names=False)
res = mod.fit()
print(res.summary())

pred = res.predict(start=train_df.index[0], end=train_df.index[-1], dynamic=False)
v_pred = res.predict(start=validate_df.index[0], end=validate_df.index[-1], dynamic=False)
f_pred = res.predict(start=forecast_df.index[0], end=forecast_df.index[-1], dynamic=False)

plt.figure(figsize=(12, 5))
plt.plot(train_df.t, train_df.x, label='Training data')
plt.plot(validate_df.t, validate_df.x, label='Validate data')
plt.plot(forecast_df.t, forecast_df.x, label='For prediction')
plt.plot(train_df.t, pred, label='Train prediction')
plt.plot(validate_df.t, v_pred, label='Validate prediction')
plt.plot(forecast_df.t, f_pred, label='Forecast prediction')
plt.legend()
plt.title('AR prediction on synthetic time series')
plt.xlabel('t')
plt.ylabel('x')
plt.grid(alpha=0.3)
plt.show()

print("Метрики на тестовой части синтетического ряда:")
print("RMSE:", np.sqrt(mean_squared_error(forecast_df.x, f_pred)))
print("MAPE:", mean_absolute_percentage_error(forecast_df.x, f_pred))
print("MAE:", mean_absolute_error(forecast_df.x, f_pred))
print("R2:", r2_score(forecast_df.x, f_pred))
print("SMAPE:", smape(forecast_df.x, f_pred))

print("\nКраткий вывод:")
print("- Исходный синтетический ряд нестационарный из-за тренда и сезонности.")
print("- После дифференцирования ряд становится стационарным.")
print("- AR-модель с лагом 12 улавливает зависимость по прошлым значениям, но качество ограничено нестационарностью исходного ряда.")


