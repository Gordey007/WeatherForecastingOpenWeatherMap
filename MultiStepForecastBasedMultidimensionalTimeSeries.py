import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import utils


# Создать DataFrame из csv файла
DF = pd.read_csv('openweathermapkomsomolskonamur20172022_2.csv')
# Вывести первые и последние 5 строк DataFrame
print(DF)

# Выбрать временные ряды.
# Временной ряд, который будет прогнозироваться должен стоять на второй позиции (clouds_all)
FEATURES_CONSIDERED = ['temp', 'clouds_all', 'leftovers']
dataset = DF[FEATURES_CONSIDERED]
# Добавить индекс (дата и время)
dataset.index = DF['dt_iso']
# Вывод полученного набора данных
print(dataset)

# График временных рядов
dataset.plot(subplots=True)
plt.show()

dataset = dataset.values

# Первые 30000 строк данных будут использоваться для обучения модели,
# оставшиеся – для её валидации
TRAIN_SPLIT = 30000
# dataset = dataset[300000:].values
# TRAIN_SPLIT = 85000

# Среднее значение
MEAN = dataset[:TRAIN_SPLIT].mean(axis=0)
# Стандартное отклонение для каждого признака
standard_deviation = dataset[:TRAIN_SPLIT].std(axis=0)

# Если стандартное отклонение равно 0, то добавить 1
i = 0
for item in standard_deviation:
    if item == 0:
        standard_deviation[i] = 1
    i += 1

# Стандартизация данных путём вычитания среднего значения
# и деления на стандартное отклонение для каждого признака
dataset = (dataset - MEAN) / standard_deviation

# Кол-во последних зарегистрированных наблюдений, которые будут подавать на вход модели
HISTORICAL_VALUES = 24
# Спрогнозировать следующие значение
FUTURE_TARGET = 5
# Установить шаг интервала
STEP = 1

# Сформировать тренировочный и валидационный набор данных для обучения
x_train_multi, y_train_multi = utils.get_multivariate_dataset(dataset, dataset[:, 1], 0,
                                                              TRAIN_SPLIT, HISTORICAL_VALUES,
                                                              FUTURE_TARGET, STEP)

x_val_multi, y_val_multi = utils.get_multivariate_dataset(dataset, dataset[:, 1],
                                                          TRAIN_SPLIT, None, HISTORICAL_VALUES,
                                                          FUTURE_TARGET, STEP)

# Преобразование датасета в тензор
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
validation_dataset = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))

BATCH_SIZE = 256
BUFFER_SIZE = 10000

# Перемешивание (shuffle), пакетирование (batch) и кэширование (cache) набора данных
train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
validation_dataset = validation_dataset.batch(BATCH_SIZE).repeat()

# Модель LSTM
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
# Установить кол-во нейронов в выходном слое
multi_step_model.add(tf.keras.layers.Dense(FUTURE_TARGET))

# Установка алгоритма оптимизации adam и функцию потерь mae
multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

# Из-за большого размера набора данных и в целях экономии времени
# каждая эпоха будет проходить только EVALUATION_INTERVAL шагов
# вместо полных данных обучения, как это обычно делается
EVALUATION_INTERVAL = TRAIN_SPLIT
VALIDATION_STEPS = EVALUATION_INTERVAL // BATCH_SIZE
# Кол-во этапов обучения (эпохи)
EPOCHS = 10

# Обучением модели
multi_step_history = multi_step_model.fit(train_dataset, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=validation_dataset,
                                          validation_steps=VALIDATION_STEPS)

# Вывод графика обучения и значение функции потерь на валидации
utils.plot_train_history(multi_step_history, 'Ошибка на обучении и валидации')

# Выполнить n кол-во прогнозов
n = 10
for x, y in validation_dataset.take(n):
    utils.multi_step_plot(x[0] * standard_deviation[1] + MEAN[1],
                    y[0] * standard_deviation[1] + MEAN[1],
                    multi_step_model.predict(x)[0] * standard_deviation[1] + MEAN[1], STEP)
