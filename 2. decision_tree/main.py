import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np


# Загрузите выборку из файла titanic.csv с помощью пакета Pandas.
data = pd.read_csv('./titanic.csv', index_col='PassengerId')

# Оставьте в выборке четыре признака: класс пассажира (Pclass), цену билета (Fare),
# возраст пассажира (Age) и его пол (Sex). Обратите внимание, что признак Sex имеет строковые значения.
data = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]
data.loc[:, 'Sex'] = data.Sex.map({'male': 1, 'female': 0})

# В данных есть пропущенные значения — например, для некоторых пассажиров неизвестен их возраст.
# Такие записи при чтении их в pandas принимают значение nan. Найдите все объекты,
# у которых есть пропущенные признаки, и удалите их из выборки.
data.dropna(inplace=True)

# Выделите целевую переменную — она записана в столбце Survived.
X = data.drop('Survived', axis=1)
y = data['Survived']

# Обучите решающее дерево с параметром random_state=241 и остальными параметрами по умолчанию
# (речь идет о параметрах конструктора DecisionTreeСlassifier).
model = DecisionTreeClassifier(random_state=241)
model.fit(X, y)

# Вычислите важности признаков и найдите два признака с наибольшей важностью.
# Их названия будут ответами для данной задачи
# (в качестве ответа укажите названия признаков через запятую или пробел, порядок не важен). 

important_feature = zip(X.columns, model.feature_importances_)
important_feature = sorted(important_feature, key=lambda f: f[1], reverse=True)
must_imported = (f[0] for f in important_feature[:2]) 

with open('answer', 'w') as f:
    f.write(' '.join(map(str, must_imported)))