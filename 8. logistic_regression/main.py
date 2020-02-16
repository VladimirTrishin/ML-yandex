# Загрузите данные из файла data-logistic.csv. Это двумерная выборка, целевая переменная на которой принимает
# значения -1 или 1.
import pandas as pd
data = pd.read_csv('data-logistic.csv', header=None)
X = data[[1, 2]]
y = data[0]

# Убедитесь, что выше выписаны правильные формулы для градиентного спуска.
# Обратите внимание, что мы используем полноценный градиентный спуск, а не его стохастический вариант!
# Реализуйте градиентный спуск для обычной и L2-регуляризованной (с коэффициентом регуляризации 10)
# логистической регрессии. Используйте длину шага k=0.1. В качестве начального приближения используйте вектор (0, 0).
import numpy as np

class Model():
    def __init__(self, c):
        self.c = c
        self.w = np.zeros(2)

    def forward(self, X):
        return 1 / (1 + np.exp(-X.dot(self.w)))

    def backward(self, X, y):
        return np.mean((X * ((self.forward(X * y[:, np.newaxis]) - 1) * y)[:, np.newaxis]), axis=0) + self.c * self.w

# Запустите градиентный спуск и доведите до сходимости (евклидово расстояние между векторами весов на
# соседних итерациях должно быть не больше 1e-5). Рекомендуется ограничить сверху число итераций десятью тысячами.
m_without_reg = Model(0)
m_with_reg = Model(10)
m = Model(0)
k = 0.1

def fit(m, k, dw_stop, e_count):
    for e in range(0, 10000):
        dw = k * m.backward(X, y)
        m.w -= dw
        if np.linalg.norm(dw) < dw_stop:
            print(e)
            break

fit(m_without_reg, k, 0.00001, 10000)
fit(m_with_reg, k, 0.00001, 10000)

print(f'{m_with_reg.w}\n{m_without_reg.w}')
# Какое значение принимает AUC-ROC на обучении без регуляризации и при ее использовании?
# Эти величины будут ответом на задание. В качестве ответа приведите два числа через пробел.
# Обратите внимание, что на вход функции roc_auc_score нужно подавать оценки вероятностей,
# подсчитанные обученным алгоритмом. Для этого воспользуйтесь сигмоидной функцией:
# a(x) = 1 / (1 + exp(-w1 x1 - w2 x2)).

from sklearn.metrics import roc_auc_score

score_with_reg = roc_auc_score((y+1)/2, m_with_reg.forward(X))
score_without_reg = roc_auc_score((y+1)/2, m_without_reg.forward(X))
score_without_fit = roc_auc_score((y+1)/2, m.forward(X))
print(f'{score_with_reg} {score_without_reg} {score_without_fit}')

with open('./1', 'w+') as f:
        f.write(f'{score_without_reg:.3f} {score_with_reg:.3f}')

# Попробуйте поменять длину шага. Будет ли сходиться алгоритм, если делать более длинные шаги?
# Как меняется число итераций при уменьшении длины шага?
# Попробуйте менять начальное приближение. Влияет ли оно на что-нибудь?