import utils
# В этом задании вам нужно подобрать оптимальное значение k для алгоритма kNN.
# Будем использовать набор данных Wine, где требуется предсказать сорт винограда,
# из которого изготовлено вино, используя результаты химических анализов.

# Выполните следующие шаги:

# Загрузите выборку Wine по адресу:
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
data = utils.load_file('wine.data', url)

# Извлеките из данных признаки и классы. Класс записан в первом столбце (три варианта),
# признаки — в столбцах со второго по последний.
# Более подробно о сути признаков можно прочитать по адресу
# https://archive.ics.uci.edu/ml/datasets/Wine (см. также файл wine.names, приложенный к заданию)

import pandas as pd

column_names = [
    'Sort',
    'Alcohol',
    'Malic acid',
    'Ash',
    'Alcalinity of ash',
    'Magnesium',
    'Total phenols',
    'Flavanoids',
    'Nonflavanoid phenols',
    'Proanthocyanins',
    'Color intensity',
    'Hue',
    'OD280/OD315 of diluted wines',
    'Proline'
]

data = [[float(w) for w in l.split(',')] for l in data.splitlines() if l]
data = pd.DataFrame(data=data, columns=column_names)
data.Sort = data.Sort.map(int)
X = data.drop('Sort', axis=1)
y = data['Sort']

# Оценку качества необходимо провести методом кросс-валидации по 5 блокам (5-fold).
# Создайте генератор разбиений, который перемешивает выборку перед формированием блоков (shuffle=True).
# Для воспроизводимости результата, создавайте генератор KFold с фиксированным параметром random_state=42.
# В качестве меры качества используйте долю верных ответов (accuracy).

from sklearn.model_selection import KFold, cross_val_score

fold_generator = KFold(n_splits=5, shuffle=True, random_state=42)

# Найдите точность классификации на кросс-валидации для метода k ближайших соседей
# (sklearn.neighbors.KNeighborsClassifier), при k от 1 до 50.
# При каком k получилось оптимальное качество?
# Чему оно равно (число в интервале от 0 до 1)?
# Данные результаты и будут ответами на вопросы 1 и 2.

from sklearn.neighbors import KNeighborsClassifier

def find_ktop_accuracy(X, y, cv, krange):
    kNN_accuracy = [
        (k, cross_val_score(KNeighborsClassifier(n_neighbors=k), X, y, cv=cv, scoring='accuracy').mean())
        for k in krange
    ]
    return max(kNN_accuracy, key=lambda t: t[1])

top_k, top_accuracy = find_ktop_accuracy(X, y, fold_generator, range(1, 51))
utils.answer('1', f'{top_k}')
utils.answer('2', f'{top_accuracy:.2f}')

# Произведите масштабирование признаков с помощью функции sklearn.preprocessing.scale.
# Снова найдите оптимальное k на кросс-валидации.
# Какое значение k получилось оптимальным после приведения признаков к одному масштабу?
# Приведите ответы на вопросы 3 и 4. Помогло ли масштабирование признаков?

import sklearn.preprocessing

X_norm = sklearn.preprocessing.scale(X)

top_k, top_accuracy = find_ktop_accuracy(X_norm, y, fold_generator, range(1, 51))
utils.answer('3', f'{top_k}')
utils.answer('4', f'{top_accuracy:.2f}')
