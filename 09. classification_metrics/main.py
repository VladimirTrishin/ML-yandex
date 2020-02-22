import utils
# Загрузите файл classification.csv. В нем записаны истинные классы объектов выборки (колонка true) и
# ответы некоторого классификатора (колонка pred).
import pandas as pd

data = pd.read_csv('classification.csv')

# Заполните таблицу ошибок классификации:

# |                    |Actual Positive	|Actual Negative|
# |Predicted Positive  |       TP       |     FP        |
# |Predicted Negative  |       FN       |     TN        |

# Для этого подсчитайте величины TP, FP, FN и TN согласно их определениям.
# Например, FP — это количество объектов, имеющих класс 0, но отнесенных алгоритмом к классу 1.
# Ответ в данном вопросе — четыре числа через пробел.

table = data.groupby(['true', 'pred']).size()
TP, FP, FN, TN = table[1, 1], table[0, 1], table[1, 0], table[0, 0]
utils.answer('1', f'{TP} {FP} {FN} {TN}')

# Посчитайте основные метрики качества классификатора:
#   Accuracy (доля верно угаданных) — sklearn.metrics.accuracy_score
#   Precision (точность) — sklearn.metrics.precision_score
#   Recall (полнота) — sklearn.metrics.recall_score
#   F-мера — sklearn.metrics.f1_score
# В качестве ответа укажите эти четыре числа через пробел.
accuracy = (TP + TN) / table.sum()
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall) 
utils.answer('2', f'{accuracy:.2} {precision:.2} {recall:.2} {f1:.2}')

# Имеется четыре обученных классификатора. В файле scores.csv записаны истинные классы и значения
# степени принадлежности положительному классу для каждого классификатора на некоторой выборке:
#   для логистической регрессии — вероятность положительного класса (колонка score_logreg),
#   для SVM — отступ от разделяющей поверхности (колонка score_svm),
#   для метрического алгоритма — взвешенная сумма классов соседей (колонка score_knn),
#   для решающего дерева — доля положительных объектов в листе (колонка score_tree).
# Загрузите этот файл.
scores = pd.read_csv('scores.csv')
# Посчитайте площадь под ROC-кривой для каждого классификатора.
# Какой классификатор имеет наибольшее значение метрики AUC-ROC (укажите название столбца)?
# Воспользуйтесь функцией sklearn.metrics.roc_auc_score.
from sklearn import metrics

classifiers = ('score_logreg', 'score_svm', 'score_knn', 'score_tree')
roc_aur = {c: metrics.roc_auc_score(scores['true'], scores[c]) for c in classifiers}
best_model, best_mark = max(roc_aur.items(), key=lambda p: p[1])
utils.answer('3', f'{best_model}')

# Какой классификатор достигает наибольшей точности (Precision) при полноте (Recall) не менее 70% ?
# Чтобы получить ответ на этот вопрос, найдите все точки precision-recall-кривой с помощью функции
# sklearn.metrics.precision_recall_curve. Она возвращает три массива: precision, recall, thresholds.
# В них записаны точность и полнота при определенных порогах, указанных в массиве thresholds.
# Найдите максимальной значение точности среди тех записей, для которых полнота не меньше, чем 0.7.

pr_curve = metrics.precision_recall_curve
pr_curves = {
    m: {'precision': curve[0], 'recall': curve[1]} 
    for m, curve in ((m, pr_curve(scores['true'], scores[m])) for m in classifiers)}

best_classifiers_precision = (
    (m, max(curve['precision'][curve['recall'] >= 0.7]))
    for m, curve in pr_curves.items())
best_classifier, best_mark = max(best_classifiers_precision, key=lambda r: r[1])
utils.answer('4', f'{best_classifier}')
