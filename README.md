# Анализ настроений с использованием наивного байесовского классификатора
classifier_bnb.py - наивный байесовский классификатор Бернулли.  
classifier_mnb.py - мультиномиальный наивный байесовский классификатор.  
classifier_best.py - классификатор, показавший наилучшую точность на dev set (после подбора гиперпараметров и экспериментов)  

## Теоретическая часть
- Наивный байесовский классификатор, модель Бернулли  
Пусть V – множество {v1,v2,...,vM} всевозможных слов (словарь). В модели Бернулли для  
каждого класса cj рассматривается случайная M-мерная переменная (B1j,B2j,...,BMj),  
компоненты которой независимы, i-ая компонента соответствует vi (i-ому слову словаря) и  
имеет распределение Бернулли с вероятностью успеха pij. Другими словами, для каждого  
класса cj у нас имеется свой набор из M несимметричных монет. Чтобы сгенерировать  
документ (множество слов) из класса cj, нужно подбросить все эти монеты и выбрать из  
словаря слова, соответствующие монетам, на которых выпал орел. Документы,  
отличающиеся только порядком слов или их частотами, считаются одинаковыми.  
В каждом из следующих вопросов выпишите сначала формулу в общем виде, а затем ее  
оценку на обучающей выборке с использованием аддитивного сглаживания (сглаживания  
Лапласа).  
1) Чему равна вероятность P(vi d | d cj) встретить i-ое слово ∈ ∈ из словаря в случайном  
документе класса cj?  
2) Вывести P(d=(k1,k2,...,kM) | d∈cj) – вероятность того, что случайный документ d,  
принадлежащий классу cj, будет состоять из k1,k2,...,kM вхождений слов v1,v2,...,vM? Для  
вывода использовать “наивное” предположение о независимости признаков.  
3) Вывести вероятность P(cj | d), что данный документ принадлежит классу cj. Для вывода  
использовать формулу Байеса.  
4) Какой класс cj будет будет выдан для документа d классификатором, если предположить,  
что P(cj) и P(d | cj) заданы? Как можно оценить вероятность ошибки?  
