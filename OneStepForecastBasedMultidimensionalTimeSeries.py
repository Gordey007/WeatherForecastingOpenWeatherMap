import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import utils


# Создать DataFrame из csv файла
DF = pd.read_csv('openweathermapkomsomolskonamur20172022_2.csv')
# Вывести первые и последние 5 строк DataFrame
print(DF)

# Выбрать временные ряды
# Прогнозирование временной ряд должен стоять на второй позиции
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
HISTORICAL_VALUES = 72
# Спрогнозировать следующие значение
FUTURE_TARGET = 24
# Установить шаг интервала
STEP = 1

# Сформировать тренировочные и валидационный наборы данных для обучения
X_TRAIN, Y_TRAIN = utils.get_multivariate_dataset(dataset, dataset[:, 1], 0,
                                                  TRAIN_SPLIT, HISTORICAL_VALUES,
                                                  FUTURE_TARGET, STEP,
                                                  single_step=True)

X_VALIDATION, Y_VALIDATION = utils.get_multivariate_dataset(dataset, dataset[:, 1],
                                                            TRAIN_SPLIT, None,
                                                            HISTORICAL_VALUES, FUTURE_TARGET,
                                                            STEP, single_step=True)

# Преобразование датасета в тензор
train_dataset = tf.data.Dataset.from_tensor_slices((X_TRAIN, Y_TRAIN))
validation_dataset = tf.data.Dataset.from_tensor_slices((X_VALIDATION, Y_VALIDATION))

BATCH_SIZE = 256
BUFFER_SIZE = 10000

# Перемешивание (shuffle), пакетирование (batch) и кэширование (cache) набора данных
train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
validation_dataset = validation_dataset.batch(BATCH_SIZE).repeat()

# Модель LSTM
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(32, input_shape=X_TRAIN.shape[-2:]))
# Установить кол-во нейронов в выходном слое
model.add(tf.keras.layers.Dense(1))

# Установка алгоритма оптимизации adam и функцию потерь mae
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

# Из-за большого размера набора данных и в целях экономии времени
# каждая эпоха будет проходить только EVALUATION_INTERVAL шагов
# вместо полных данных обучения, как это обычно делается
EVALUATION_INTERVAL = TRAIN_SPLIT
# Кол-во этапов обучения (эпохи)
EPOCHS = 10
VALIDATION_STEPS = EVALUATION_INTERVAL // BATCH_SIZE

# Обучением модели
HISTORY_TRAINING = model.fit(train_dataset, epochs=EPOCHS,
                             steps_per_epoch=EVALUATION_INTERVAL,
                             validation_data=validation_dataset,
                             validation_steps=VALIDATION_STEPS)

# Вывод графика одношагового обучения и
# значение функции потерь на валидационном датасете
utils.plot_train_history(HISTORY_TRAINING, 'Ошибка на обучении и валидации')

# Выполнить n кол-во одношаговых прогнозов
n = 10
for x, y in validation_dataset.take(n):
    plot = utils.show_plot([x[0][:, 1].numpy(), y[0].numpy(),
                            model.predict(x)[0]], 12,
                           'Точечное прогнозирование на основе многомерного временного ряда',
                           standard_deviation[1], MEAN[1])
    plot.show()
