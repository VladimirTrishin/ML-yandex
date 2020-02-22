import utils
# Мы будем использовать в данном задании набор данных Boston,
# где нужно предсказать стоимость жилья на основе различных характеристик расположения
# (загрязненность воздуха, близость к дорогам и т.д.).
# Подробнее о признаках можно почитать по адресу:
# https://archive.ics.uci.edu/ml/machine-learning-databases/housing/

# Загрузите выборку Boston с помощью функции sklearn.datasets.load_boston().
# Результатом вызова данной функции является объект, у которого признаки записаны в поле data,
# а целевой вектор — в поле target.
import sklearn.datasets as datasets

X, y = datasets.load_boston(True)

# Приведите признаки в выборке к одному масштабу при помощи функции sklearn.preprocessing.scale.
from sklearn import preprocessing
X = preprocessing.scale(X)

# Переберите разные варианты параметра метрики p по сетке от 1 до 10 с таким шагом,
# чтобы всего было протестировано 200 вариантов (используйте функцию numpy.linspace).
# Используйте KNeighborsRegressor с n_neighbors=5 и weights='distance' — 
# данный параметр добавляет в алгоритм веса, зависящие от расстояния до ближайших соседей.
# В качестве метрики качества используйте среднеквадратичную ошибку
# (параметр scoring='mean_squared_error' у cross_val_score;
# при использовании библиотеки scikit-learn версии 0.18.1 и выше необходимо указывать
# scoring='neg_mean_squared_error').
# Качество оценивайте, как и в предыдущем задании, с помощью кросс-валидации по 5 блокам с random_state = 42,
# не забудьте включить перемешивание выборки (shuffle=True).
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

cv = KFold(n_splits=5, shuffle=True, random_state=42)
prange = np.linspace(1, 10, 200)

def model(p):
    return KNeighborsRegressor(n_neighbors=5,weights='distance', metric='minkowski', p=p) 

mean_sqr_error = [
    (p, cross_val_score(model(p), X, y, cv=cv, scoring='neg_mean_squared_error').mean())
    for p in prange
]

# Определите, при каком p качество на кросс-валидации оказалось оптимальным.
# Обратите внимание, что cross_val_score возвращает массив показателей качества по блокам;
# необходимо максимизировать среднее этих показателей. Это значение параметра и будет ответом на задачу.
# Если ответом является нецелое число, то целую и дробную часть необходимо разграничивать точкой,
# например, 0.4. При необходимости округляйте дробную часть до одного знака.

best_p, best_error = max(mean_sqr_error, key=lambda m: m[1])
print(mean_sqr_error)
utils.answer('1', f'{best_p:.1f}')
