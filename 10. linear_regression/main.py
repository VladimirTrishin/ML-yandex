import utils
# Загрузите данные об описаниях вакансий и соответствующих годовых зарплатах из файла salary-train.csv
# (либо его заархивированную версию salary-train.zip).
import pandas as pd

data = pd.read_csv('salary-train.csv')

# Проведите предобработку:
# Приведите тексты к нижнему регистру (text.lower()).
data['FullDescription'] = data['FullDescription'].str.lower()

# Замените все, кроме букв и цифр, на пробелы — это облегчит дальнейшее разделение текста на слова.
# Для такой замены в строке text подходит следующий вызов: re.sub('[^a-zA-Z0-9]', ' ', text). 
# Также можно воспользоваться методом replace у DataFrame, чтобы сразу преобразовать все тексты.
data['FullDescription'] = data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

# Примените TfidfVectorizer для преобразования текстов в векторы признаков.
# Оставьте только те слова, которые встречаются хотя бы в 5 объектах (параметр min_df у TfidfVectorizer).
from sklearn.feature_extraction.text import TfidfVectorizer

enc = TfidfVectorizer(min_df=5)

X = enc.fit_transform(data.FullDescription)
# Замените пропуски в столбцах LocationNormalized и ContractTime на специальную строку 'nan'.
data.fillna('nan', inplace=True)

# Примените DictVectorizer для получения one-hot-кодирования признаков LocationNormalized и ContractTime.
from sklearn.feature_extraction import DictVectorizer

one_hot = DictVectorizer()
X_categ = one_hot.fit_transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))

# Объедините все полученные признаки в одну матрицу "объекты-признаки".
# Обратите внимание, что матрицы для текстов и категориальных признаков являются разреженными.
# Для объединения их столбцов нужно воспользоваться функцией scipy.sparse.hstack.
import scipy

X = scipy.sparse.hstack([X, X_categ])

# Обучите гребневую регрессию с параметрами alpha=1 и random_state=241.
# Целевая переменная записана в столбце SalaryNormalized.
from sklearn.linear_model import Ridge

y = data.SalaryNormalized
model = Ridge(alpha=1, random_state=241)
model.fit(X, y)

# Постройте прогнозы для двух примеров из файла salary-test-mini.csv.
# Значения полученных прогнозов являются ответом на задание. Укажите их через пробел
data_test = pd.read_csv('salary-test-mini.csv')
data_test['FullDescription'] = data_test['FullDescription'].str.lower()
data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
X_test = enc.transform(data_test.FullDescription)
data_test.fillna('nan', inplace=True)
X_test_categ = one_hot.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = scipy.sparse.hstack([X_test, X_test_categ])
predicted = model.predict(X_test)

utils.answer('1', ' '.join(format(p, '.2f') for p in predicted))