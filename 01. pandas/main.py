import pandas as pd
import numpy as np

# Необходимо ответить на следующие вопросы:
def question_N1(data):
    """
    Какое количество мужчин и женщин ехало на корабле? 
    В качестве ответа приведите два числа через пробел.
    """
    gender_count = data.Sex.value_counts()
    return f"{gender_count['male']} {gender_count['female']}"


def question_N2(data):
    """
    Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров.
    Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен), округлив до двух знаков.
    """
    survives_pc = data.Survived.mean() * 100
    return f'{survives_pc:.2f}'


def question_N3(data):
    """
    Какую долю пассажиры первого класса составляли среди всех пассажиров?
    Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен), округлив до двух знаков.
    """
    first_class_pc = data.Pclass.value_counts()[1] / data.Pclass.count() * 100
    return f'{first_class_pc:.2f}'


def question_N4(data):
    """
    Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров.
    В качестве ответа приведите два числа через пробел.
    """
    return f'{data.Age.mean():.2f} {data.Age.median():.2f}'


def question_N5(data):
    """
    Коррелируют ли число братьев/сестер/супругов с числом родителей/детей?
    Посчитайте корреляцию Пирсона между признаками SibSp и Parch.
    """
    return f"{data['SibSp'].corr(data['Parch'], method='pearson'):.2f}"


def question_N6(data):
    """
    Какое самое популярное женское имя на корабле?
    Извлеките из полного имени пассажира (колонка Name) его личное имя (First Name).
    Это задание — типичный пример того, с чем сталкивается специалист по анализу данных.
    Данные очень разнородные и шумные, но из них требуется извлечь необходимую информацию.
    Попробуйте вручную разобрать несколько значений столбца Name и выработать правило для извлечения имен,
    а также разделения их на женские и мужские.
    """
    
    def split_name(full_name):
        #TODO: Corret parse maried names
        return [] if 'Mrs.' in full_name else \
        [n for n in full_name.split() if n != 'Miss.']

    split_names = data[data.Sex == 'female'].Name.apply(split_name).values
    all_names = np.concatenate(split_names)
    names, counts = np.unique(all_names, return_counts=True)
    sorted_names = names[np.argsort(counts)]    
    return sorted_names[-1]


def print_to_file(file_name, s):
    with open(file_name, "w") as f:
        f.write(s)


data = pd.read_csv('titanic.csv', index_col='PassengerId')

for i in range(1, 7):
    print_to_file(f'./answers/{i}', globals()[f'question_N{i}'](data))