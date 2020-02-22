import utils

# Загрузите объекты из новостного датасета 20 newsgroups, относящиеся к категориям "космос" и "атеизм"
# (инструкция приведена выше). Обратите внимание, что загрузка данных может занять несколько минут...
from sklearn import datasets

newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
X = newsgroups.data
y = newsgroups.target

# Вычислите TF-IDF-признаки для всех текстов. Обратите внимание, что в этом задании мы предлагаем вам
# вычислить TF-IDF по всем данным. При таком подходе получается, что признаки на обучающем множестве
# используют информацию из тестовой выборки — но такая ситуация вполне законна, поскольку мы не используем значения
# целевой переменной из теста. На практике нередко встречаются ситуации, когда признаки объектов тестовой выборки
# известны на момент обучения, и поэтому можно ими пользоваться при обучении алгоритма.

from sklearn.feature_extraction.text import TfidfVectorizer

feature_extractor = TfidfVectorizer()
X = feature_extractor.fit_transform(X)

# Подберите минимальный лучший параметр C из множества [10^-5, 10^-4, ... 10^4, 10^5] для SVM с линейным ядром
# (kernel='linear') при помощи кросс-валидации по 5 блокам. Укажите параметр random_state=241 и для SVM, и для KFold.
# В качестве меры качества используйте долю верных ответов (accuracy).

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
import numpy as np

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
model = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(model, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)
best_C = gs.best_params_['C']

# Обучите SVM по всей выборке с оптимальным параметром C, найденным на предыдущем шаге.
# Найдите 10 слов с наибольшим абсолютным значением веса (веса хранятся в поле coef_ у svm.SVC).
# Они являются ответом на это задание. Укажите эти слова через запятую или пробел, в нижнем регистре,
# в лексикографическом порядке.

model = SVC(kernel='linear', C=best_C)
model.fit(X, y)

best_ten_feature = np.argsort(-np.abs(model.coef_.toarray()))[0, :10]
feature_names = feature_extractor.get_feature_names()
best_ten_feature = sorted([feature_names[f] for f in best_ten_feature])

utils.answer('1', ' '.join(best_ten_feature))