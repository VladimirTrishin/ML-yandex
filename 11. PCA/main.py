import utils
# Загрузите данные close_prices.csv. В этом файле приведены цены акций 30 компаний на закрытии
# торгов за каждый день периода.
import pandas as pd

data = pd.read_csv('close_prices.csv', index_col='date')
# На загруженных данных обучите преобразование PCA с числом компоненты равным 10.
# Скольких компонент хватит, чтобы объяснить 90% дисперсии?
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
pca.fit(data)

enough = 0
total = 0
while total < 0.9:
    total += pca.explained_variance_ratio_[enough] 
    enough += 1
utils.answer('1', f'{enough}')

# Примените построенное преобразование к исходным данным и возьмите значения первой компоненты.
# Загрузите информацию об индексе Доу-Джонса из файла djia_index.csv.
# Чему равна корреляция Пирсона между первой компонентой и индексом Доу-Джонса?
X = pd.DataFrame(pca.transform(data), index=data.index)
djia_index = pd.read_csv('djia_index.csv', index_col='date')
corr = djia_index.corrwith(X[0])[0]
utils.answer('2', f'{corr:.2f}')

# Какая компания имеет наибольший вес в первой компоненте? Укажите ее название с большой буквы.
import numpy as np

company = data.columns[np.argmax(pca.components_[0])]
utils.answer('3', company)