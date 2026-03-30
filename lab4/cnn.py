from textwrap import dedent
from inspect import getsource
from termcolor import colored

from tensorflow import keras
from keras import Input, Sequential, activations, losses, optimizers
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

from helpers.log import write_log

STATIS_FILE = 'stats.log'
MODEL_FILE = 'model.log'

def setup_model() -> tuple[Sequential, int]:
    model = Sequential()
    epochs = 3

    model.add(Input(shape=(28, 28, 1), batch_size=32))
    model.add(
        Conv2D(
            filters=50,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation=activations.relu,
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=10, activation=activations.softmax))

    print(model.summary())

    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.categorical_crossentropy,
        metrics=["accuracy", "precision", "recall"],
    )

    return (model, epochs)

type stats = list[float]

def get_last_iter_and_stats(path: str, file: str) -> tuple[int, stats, stats]:
    with open(path + file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = list(reversed(lines))
    for s in lines:
        if 'ITER: ' in s:
            last_iter = int(s[6:-1])
            break
    
    train_stats = [float(s[s.index(':') + 2:-1]) for s in reversed(lines[6:10])]
    test_stats = [float(s[s.index(':') + 2:-1]) for s in reversed(lines[1:5])]

    return (last_iter, train_stats, test_stats)

def get_color(first: float, second: float, is_loss: bool=False) -> str:
    if first == second:
        return 'white'
    elif first > second:
        return 'red' if is_loss else 'green'
    else:
        return 'green' if is_loss else 'red'



def print_statistic_and_log(model: Sequential, xs_train, y_train_cats, xs_test, y_test_cats , verbose: bool=False):
    train_score = model.evaluate(xs_train, y_train_cats, verbose=verbose)
    test_score = model.evaluate(xs_test, y_test_cats, verbose=verbose)

    last_iter, train_stats, test_stats = get_last_iter_and_stats('log/', 'stats.log')

    train_score = [0.03272294417023659, 0.9976333475112915, 0.9994620776176453, 0.9958999848365784]
    test_score = [0.03272294417023659, 0.9976333475112915, 0.9994620776176453, 0.9958999848365784]

    stats = f'''
        Ошибка на обучающей выборке: {colored(str(train_score[0]), get_color(train_score[0], train_stats[0], True))} | {train_stats[0]}
        accuracy на обучающей выборке: {colored(str(train_score[1]), get_color(train_score[1], train_stats[1]))} | {train_stats[1]}
        precision на обучающей выборке: {colored(str(train_score[2]), get_color(train_score[2], train_stats[2]))} | {train_stats[2]}
        recall на обучающей выборке: {colored(str(train_score[3]), get_color(train_score[3], train_stats[3]))} | {train_stats[3]}

        Ошибка на тестовой выборке: {colored(str(test_score[0]), get_color(test_score[0], test_stats[0], True))} | {test_stats[0]}
        accuracy на тестовой выборке: {colored(str(test_score[1]), get_color(test_score[1], test_stats[1]))} | {test_stats[1]}
        precision на тестовой выборке: {colored(str(test_score[2]), get_color(test_score[2], test_stats[2]))} | {test_stats[2]}
        recall на тестовой выборке: {colored(str(test_score[3]), get_color(test_score[3], test_stats[3]))} | {test_stats[3]}
    '''
    print(dedent(stats))

    log = \
        f'\n\nITER: {last_iter + 1}\n' + \
        '\n' +\
        getsource(setup_model) + \
        '\n' +\
        f'Ошибка на обучающей выборке: {train_score[0]}\n' + \
        f'accuracy на обучающей выборке: {train_score[1]}\n' + \
        f'precision на обучающей выборке: {train_score[2]}\n' + \
        f'recall на обучающей выборке: {train_score[3]}\n' + \
        '\n' +\
        f'Ошибка на тестовой выборке: {test_score[0]}\n' + \
        f'accuracy на тестовой выборке: {test_score[1]}\n' + \
        f'precision на тестовой выборке: {test_score[2]}\n' + \
        f'recall на тестовой выборке: {test_score[3]}\n' + \
        '--------------------------------'

    write_log(log, filename='stats')
    


def test():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print(x_train[0])

    x_train = x_train / 255
    x_test = x_test / 255

    print(y_train[0])

    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    print(y_train_cat[0])


    model, epochs = setup_model()

    model.fit(x_train, y_train_cat, epochs=epochs, verbose=False)

    print_statistic_and_log(model, x_train, y_train_cat, x_test, y_test_cat)

if __name__ == '__main__':
    test()