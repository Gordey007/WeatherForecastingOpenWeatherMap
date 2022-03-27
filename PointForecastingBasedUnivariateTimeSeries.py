import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import utils


# Создать DataFrame из csv файла
DF = pd.read_csv('openweathermapkomsomolskonamur20172022_2.csv')
# Вывести первые и последние 5 строк DataFrame
print(DF)

# Для обеспечения воспроизводимости результатов
tf.random.set_seed(13)

# Взять значения из DataFrame который будут прогнозироваться
dataset = DF['clouds_all']
# Добавить индекс (дата и время)
dataset.index = DF['dt_iso']
# Вывести первые и последние 5 строк
print(dataset)

# Вывести временной ряд, который будет прогнозироваться
dataset.plot(subplots=True)
plt.show()

dataset = dataset.values

# Первые 30000 строк данных будут использоваться для обучения модели,
# оставшиеся – для её валидации
TRAIN_SPLIT = 30000
# dataset = dataset[300000:].values
# TRAIN_SPLIT = 85000

# Среднее значение
MEAN = dataset[:TRAIN_SPLIT].mean()
# Стандартное отклонение для каждого признака
STANDARD_DEVIATION = dataset[:TRAIN_SPLIT].std()
# Стандартизация данных путём вычитания среднего значения
# и деления на стандартное отклонение для каждого признака
dataset = (dataset - MEAN) / STANDARD_DEVIATION

# Кол-во последних зарегистрированных наблюдений, которые будут подавать на вход модели
HISTORICAL_VALUES = 10
# Спрогнозировать следующие значение
FUTURE_TARGET = 0

# Сформировать наборы данных для обучения
# X_TRAIN = [наблюдения, временной интервал, кол-во признаков]
X_TRAIN, Y_TRAIN = utils.get_dataset(dataset, 0, TRAIN_SPLIT, HISTORICAL_VALUES,
                                     FUTURE_TARGET)

X_VALIDATION, Y_VALIDATION = utils.get_dataset(dataset, TRAIN_SPLIT, None, HISTORICAL_VALUES,
                                               FUTURE_TARGET)

# Преобразование датасета в тензор
train_dataset = tf.data.Dataset.from_tensor_slices((X_TRAIN, Y_TRAIN))
validation_dataset = tf.data.Dataset.from_tensor_slices((X_VALIDATION, Y_VALIDATION))

BUFFER_SIZE = 10000
BATCH_SIZE = 256

# Перемешивание (shuffle), пакетирование (batch) и кэширование (cache) набора данных
train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
validation_dataset = validation_dataset.batch(BATCH_SIZE).repeat()

# Модель LSTM
model = tf.keras.models.Sequential([tf.keras.layers.LSTM(8, input_shape=X_TRAIN.shape[-2:]),
                                    # Установить кол-во нейронов в выходном слое
                                    tf.keras.layers.Dense(1)
                                    ])

# Установка алгоритма оптимизации adam и функцию потерь mae
model.compile(optimizer='adam', loss='mae')

# Из-за большого размера набора данных и в целях экономии времени
# каждая эпоха будет проходить только EVALUATION_INTERVAL шагов
# вместо полных данных обучения, как это обычно делается
EVALUATION_INTERVAL = TRAIN_SPLIT
# Кол-во этапов обучения (эпохи)
EPOCHS = 10
VALIDATION_STEPS = EVALUATION_INTERVAL // BATCH_SIZE

# Обучением модели
model.fit(train_dataset, epochs=EPOCHS,
          steps_per_epoch=EVALUATION_INTERVAL,
          validation_data=validation_dataset, validation_steps=VALIDATION_STEPS)

# Выполнить n кол-во одношаговых прогнозов
n = 10
for x, y in validation_dataset.take(n):
    plot = utils.show_plot([x[0].numpy(), y[0].numpy(),
                            model.predict(x)[0]], 0, 'Точечное прогнозирование на основе '
                                                     'одного временного ряда',
                           STANDARD_DEVIATION, MEAN)
    plot.show()
