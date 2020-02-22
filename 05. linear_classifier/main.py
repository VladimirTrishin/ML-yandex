import utils

# Инструкция по выполнению
# Загрузите обучающую и тестовую выборки из файлов perceptron-train.csv и perceptron-test.csv.
# Целевая переменная записана в первом столбце, признаки — во втором и третьем.
import pandas as pd

train = pd.read_csv('perceptron-train.csv', header=None)
test = pd.read_csv('perceptron-test.csv', header=None)

X_train, y_train = train[[1, 2]], train[0]
X_test, y_test = test[[1, 2]], test[0]


# Обучите персептрон со стандартными параметрами и random_state=241.
# Подсчитайте качество (долю правильно классифицированных объектов, accuracy) полученного классификатора
# на тестовой выборке.
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

model = Perceptron(random_state=241)
model.fit(X_train, y_train)
unnorm_accuracy = accuracy_score(y_test, model.predict(X_test))
# Нормализуйте обучающую и тестовую выборку с помощью класса StandardScaler.
# Обучите персептрон на новой выборке. Найдите долю правильных ответов на тестовой выборке.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)
model = Perceptron()
model.fit(X_train_norm, y_train)
norm_accuracy = accuracy_score(y_test, model.predict(X_test_norm))

# Найдите разность между качеством на тестовой выборке после нормализации и качеством до нее.
# Это число и будет ответом на задание.
# Если ответом является нецелое число, то целую и дробную часть необходимо разграничивать точкой,
# например, 0.421. При необходимости округляйте дробную часть до ТРЕХ знаков.

diff_accuracy = norm_accuracy - unnorm_accuracy
print(f'{norm_accuracy} {unnorm_accuracy}')
utils.answer('1', f'{diff_accuracy:.3f}')