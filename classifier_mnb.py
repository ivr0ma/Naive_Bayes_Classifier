#import numpy as np
#import matplotlib.pyplot as plt
import copy
import os
import math
from math import log

#=======================================================================================================================
#--------------------------Обработка документов(строк). Отделяет символы от слов----------------------------------------
#=======================================================================================================================
def Preparing(mas):
    sym = '.,:;&()<>!?"-'
    for i in range(len(mas)):
        p = ''
        for j in range(len(mas[i])):
            if mas[i][j] in sym:
                mas.append(mas[i][j])
            else:
                p += mas[i][j]
        if p != '':
            mas[i] = p
    return mas

#=======================================================================================================================
#------------------------------------------------Загрузка train---------------------------------------------------------
#=======================================================================================================================
def LoadTrain():
    TrainDocs = []    # -- массив документов
    TrainLabels = []  # -- массив меток

    with open('train.texts', encoding='utf-8') as Fdocs:
        with open('train.labels', encoding='utf-8') as Flabels:
            for line in Fdocs:
                doc = line                  # -- считываем документ(строку)
                label = Flabels.readline()  # -- считываем метку
                TrainDocs.append(doc)
                TrainLabels.append(label[:-1])

    #print(PosDocs[-1][-1])
    #print(NegDocs[-1][-1])

    return TrainDocs, TrainLabels

#=======================================================================================================================
#------------------------------------------------Загрузка текста--------------------------------------------------------
#=======================================================================================================================
def LoadT(s):
    Docs = []    # -- массив документов

    with open(s, encoding='utf-8') as Fdocs:
        for line in Fdocs:
            doc = line  # -- считываем документ(строку)
            Docs.append(doc)

    #print(PosDocs[-1][-1])
    #print(NegDocs[-1][-1])

    return Docs

#=======================================================================================================================
#-----------------------------------------------Загрузка меток----------------------------------------------------------
#=======================================================================================================================
def LoadL(s):
    Docs = []  # -- массив документов

    with open(s, encoding='utf-8') as Fdocs:
        for line in Fdocs:
            doc = line  # -- считываем документ(строку)
            doc = doc[:-1]
            Docs.append(doc)

    # print(PosDocs[-1][-1])
    # print(NegDocs[-1][-1])

    return Docs

#=======================================================================================================================
#----------------------------------------------Печать статистики--------------------------------------------------------
#=======================================================================================================================
def PrintLen(Docs):
    DataLen = 0
    MasLen = []
    for doc in Docs:
        LenDoc = 0
        for wrd in doc:
            LenDoc += len(wrd)
        DataLen += LenDoc
        MasLen.append(LenDoc)
    MasLen.sort()
    m = len(MasLen)
    print('Наименьшая длина: ', MasLen[0])
    print('Наибольшая длина: ', MasLen[m - 1])
    print('Средняя длина: ', DataLen / m)
    if m % 2 == 0:
        print('Медианная длина: ', (MasLen[m // 2] + MasLen[m // 2 - 1]) / 2)
    else:
        print('Медианная длина: ', MasLen[m // 2])

#=======================================================================================================================
#----------------------------------------Обучение. Подсчет вероятностей-------------------------------------------------
#=======================================================================================================================
def train(train_texts, train_labels):
    # ==================================================================================================================
    # -- Создание двух списков pos и neg -------------------------------------------------------------------------------
    # ==================================================================================================================
    def CrList(train_texts, train_labels):
        PosDocs = []
        NegDocs = []
        N = len(train_labels)
        print('===',N)
        for i in range(N):
            doc = train_texts[i]
            doc = doc.lower()  # -- перевод в нижний регистр
            doc = doc.replace('<br />', ' ')  # -- убираем символ <br />
            doc = doc.split()  # -- разбиваем на слова
            doc = Preparing(doc)

            if train_labels[i] == 'pos':
                PosDocs.append(doc)
            elif train_labels[i] == 'neg':
                NegDocs.append(doc)
        return PosDocs, NegDocs

    # ==================================================================================================================
    # -- Создание двух словарей слово->частота -------------------------------------------------------------------------
    # ==================================================================================================================
    def CrDicts(PosDocs, NegDocs):
        PosSet = set()
        NegSet = set()
        for doc in PosDocs:
            PosSet = PosSet.union(set(doc))
        for doc in NegDocs:
            NegSet = NegSet.union(set(doc))
        # print(PosSet)
        PosDict = {wrd: 0 for wrd in PosSet}
        NegDict = {wrd: 0 for wrd in NegSet}

        return PosDict, NegDict

    # ==================================================================================================================
    # -- Подсчет вероятностей -------------------------------------------------------------------------
    # ==================================================================================================================
    def FindProb(DictW, Docs):
        LenAllWord = 0

        for doc in Docs:
            LenAllWord += len(doc)

        counter = 0
        LenDocs = len(Docs)
        for doc in Docs:
            for wrd in doc:
                DictW[wrd] += 1
            counter += 1
            #if counter % 100 == 0:
                #print(counter, '/', LenDocs)

        print(LenDocs, '/', LenDocs)
        for wrd in DictW:
            DictW[wrd] = DictW[wrd] / LenAllWord

    PosDocs, NegDocs = CrList(train_texts, train_labels)   # -- заполняем два списка
    PosDict, NegDict = CrDicts(PosDocs, NegDocs)           # -- заполняем два словаря
    print('созданы 2 словаря')
    print('pos:')
    PrintLen(PosDocs)
    print('neg:')
    PrintLen(NegDocs)

    print('процесс обучения...')
    FindProb(PosDict, PosDocs)       # -- находим вероятности Р(Di|pos)
    FindProb(NegDict, NegDocs)       # -- находим вероятности Р(Di|neg)
    #print(PosDict)
    # print(NegDict)

    return PosDict, NegDict, PosDocs, NegDocs

#=======================================================================================================================
#------------------------------------------------Классификация----------------------------------------------------------
#=======================================================================================================================
def classify(texts, PosDict, NegDict, PosDocs, NegDocs):
    # ==================================================================================================================
    # -- Подсчет вероятности (мультиномиальный классификатор) ----------------------------------------------------------
    # ==================================================================================================================
    def Multi(PosDict, NegDict, doc, PosProb, NegProb, AllPos, AllNeg):
        posver = log(PosProb)
        for wrd in doc:
            if wrd in PosDict:
                posver += log(PosDict[wrd])
            else:
                posver += log(1 / (1 + AllPos))
        negver = log(NegProb)
        for wrd in doc:
            if wrd in NegDict:
                negver += log(NegDict[wrd])
            else:
                negver += log(1 / (1 + AllNeg))
        if negver > posver:
            return 'neg'
        else:
            return 'pos'

    AllPos = 0
    AllNeg = 0
    for doc in PosDocs:
        AllPos += len(doc)
    for doc in NegDocs:
        AllNeg += len(doc)

    PosProb = len(PosDocs) / (len(PosDocs) + len(NegDocs))  # -- P(pos)
    NegProb = 1 - PosProb                                   # -- P(neg)

    counter = 0
    LenTexts = len(texts)
    #print('===', LenTexts)
    label = []
    for line in texts:
        res = ''
        counter += 1
        doc = line
        doc = doc.lower()
        doc = doc.replace('<br />', ' ')
        doc = doc.split()  # -- разбиваем на слова
        doc = Preparing(doc)
        res = Multi(PosDict, NegDict, doc, PosProb, NegProb, AllPos, AllNeg)
        label.append(res)
        if counter % 100:
            print(counter, '/', LenTexts)

    return label

#=======================================================================================================================
#----------------------------------------------Основная программа-------------------------------------------------------
#=======================================================================================================================
TrainDocs, TrainLabels = LoadTrain()         # -- загружаем данные
print('загрузка данных завершена')
PosDict, NegDict, PosDocs, NegDocs = train(TrainDocs, TrainLabels)
print('обучение закончилось')
print('проверка точности:')

DevDocs = LoadT('dev.texts')
print('+++',len(DevDocs))

#print(len(PosDict))

ResLabels = classify(DevDocs, PosDict, NegDict, PosDocs, NegDocs)

DevLabels = LoadL('dev.labels')
print(DevLabels)
Pright = 0
Pall = 0
for i in range(len(DevDocs)):
    if ResLabels[i] == DevLabels[i]:
        Pright += 1
    Pall += 1
print(Pright/Pall)

#print(len(DevLabels))
#print(len(ResLabels))