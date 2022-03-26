import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['figure.figsize'] = (10, 5)
mpl.rcParams['axes.grid'] = False


# Функция возвращает временные интервалы для обучения модели.
# Аргумент history_size — это размер последнего временного интервала,
# target_size – аргумент, определяющий насколько далеко в будущее модель должна
# научиться прогнозировать. Это целевой вектор, который необходимо спрогнозировать.
def get_dataset(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Изменить форму данных с (history_size,) на (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])
    return np.array(data), np.array(labels)


def create_time_steps(length):
    return list(range(-length, 0))


# Функция построения графика
def show_plot(plot_data, delta, title, standard_deviation, mean):
    labels = ['Исторические значения', 'Будущие значение', 'Предсказанное значение']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        destructuring = plot_data[i] * standard_deviation + mean
        if i:
            # print(plot_data[i])
            # print(destructuring)
            plt.plot(future, destructuring, marker[i], markersize=10, label=labels[i])
        else:
            # print(plot_data[i])
            # print(destructuring)
            plt.plot(time_steps, destructuring.flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel('Временной шаг')
    return plt


# Функция выполняет задачу организации временных интервалов с отбром
# последних наблюдений на основе заданного размера шага
def get_multivariate_dataset(dataset, target, start_index, end_index,
                             history_size, target_size, step,
                             single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)


# Функция создания графика обучения
def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Ошибка на обучении')
    plt.plot(epochs, val_loss, 'r', label='Ошибка на валидации')
    plt.title(title)
    plt.legend()

    plt.show()


# Функция визуализации
def multi_step_plot(history, true_future, prediction, step):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), label='Исторические значения')
    # print('true_future = ', true_future)
    plt.plot(np.arange(num_out) / step, np.array(true_future), 'bo', label='Будущие значение')

    if prediction.any():
        # print('prediction = ', prediction)
        plt.plot(np.arange(num_out) / step, np.array(prediction), 'ro',
                 label='Предсказанное значение')

    plt.legend(loc='upper left')
    plt.show()
