# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 23:55:45 2020

@author: Troshin Mikhail
m.troshin.ml@gmail.com
"""

import pandas as pd
import numpy as np
from exploring import if_for_regr, make_feature_categories
from preprocessing import NumPreprocessor, LabelPreprocessor, TextPreprocessor

from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV


class AutoClassifier():
    """
    Автоматический классификатор
    """
    def __init__(self):
        self.scores = {}
        self.models = []
        
    def transform_to_fit(self, X):
        # разбиваем всю матрицу объекты-признаки на категории для дальнейшей предобработки
        self._categories = make_feature_categories(X)
        to_concat = []
        
        # предобработка числовых признаков
        if self._categories['num']:
            self._num_prep = NumPreprocessor(feature_selection=False)
            num_X = self._num_prep.fit_transform(X[self._categories['num']])
            to_concat.append(num_X)
            
        # предобработка категориальных признаков и лейблов
        if (self._categories['cat'] or
            self._categories['labels']):
            self._lbl_prep = LabelPreprocessor(feature_selection=False)
            lbl_X = self._lbl_prep.fit_transform(X[(self._categories['cat'] + self._categories['labels'])])
            to_concat.append(lbl_X)
            
        # предобработка текстов
        if self._categories['text']:
            self._txt_prep = TextPreprocessor()
            txt_X = self._txt_prep.fit_transform(X[self._categories['text']])
            to_concat.append(txt_X)
            
        # сборка пространства признаков
        self.train_data = pd.concat(to_concat, axis=1)
        return self.train_data
    
    
    def transform_to_pred(self, to_pred):
        to_concat = []
        # предобработка числовых признаков
        if self._categories['num']:
            num_pred = self._num_prep.transform(to_pred[self._categories['num']])
            to_concat.append(num_pred)
            
        # предобработка категориальных признаков и лейблов
        if (self._categories['cat'] or
            self._categories['labels']):
            lbl_pred = self._lbl_prep.transform(to_pred[(self._categories['cat'] + self._categories['labels'])])
            to_concat.append(lbl_pred)
            
        # предобработка текстов
        if self._categories['text']:
            txt_pred = self._txt_prep.transform(to_pred[self._categories['text']])
            to_concat.append(txt_pred)
            
        # сборка пространства признаков
        self.pred_data = pd.concat(to_concat, axis=1)
        return self.pred_data
        
    
    def fit(self, X, y):
        """
        Метод для поиска (выбора из двух) и обучения алгоритма классификации

        Parameters
        ----------
        X : pd.DataFrame
            DESCRIPTION.
        y : pd.DataFrame
            DESCRIPTION.

        Raises
        ------
        Warning
            DESCRIPTION.

        Returns pd.DataFrame
        -------
        None.

        """
        self.X = X
        self.y = y
        
        # проверка целевой переменной
        if (if_for_regr(y)):
            raise Warning('Значения целевой переменной лучше подходят для задачи регрессии. Проверьте корректность выбора класса')
        
        # подготовка данных
        self.X  = self.transform_to_fit(self.X)
        
        # Простой линейный классификатор для бенчмарка
        from sklearn.linear_model import LogisticRegression
        cv = KFold(n_splits=5, shuffle=True, random_state=14)
        self.logreg = LogisticRegression(penalty='l2', random_state = 14)
        logreg_score = cross_val_score(self.logreg, self.X, self.y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        self.scores['logreg'] = np.mean(logreg_score)
        
        # настройка бустинга для лучшего качества
        from lightgbm import LGBMClassifier
        # GridSearch
        parameters_lgbm = [{'n_estimators': [100, 200, 500],
                            'num_leaves': [31],
                            'max_depth': [5, 10, 20, 50],
                            'learning_rate': [0.1 ,0.01],
                            'n_jobs': [3]}]
        lgbm_grid_search = GridSearchCV(estimator=LGBMClassifier(),
                                   param_grid = parameters_lgbm,
                                   scoring = 'accuracy',
                                   cv = 10,
                                   n_jobs = 3)
        lgbm_grid_search = lgbm_grid_search.fit(self.X, self.y)
        self.scores['lgbm'] = lgbm_grid_search.best_score_
        self.best_lgbm_params = lgbm_grid_search.best_params_
        
        print('LogisticRegression algorithm accuracy is %0.2f' %self.scores['logreg'])
        print('Best accuracy is %0.2f' %(max(self.scores['logreg'], self.scores['lgbm'])))
        
        # запоминаем лучшую модель
        if self.scores['lgbm'] > self.scores['logreg']:
            self.best_model = LGBMClassifier(**self.best_lgbm_params)
        else:
            self.best_model = LogisticRegression(penalty='l2')
            
        # обучаем лучшую модель
        self.best_model.fit(self.X, self.y)
        
    
    def predict(self, to_pred):
        """
        Метод для получения предсказания наилучшей обученной моделью

        Parameters
        ----------
        to_pred : TYPE
            DESCRIPTION.

        Returns
        -------
        predicted : TYPE
            DESCRIPTION.

        """
        
        self.to_pred = to_pred
        
        # Преобразование данных для предсказания
        self.to_pred = self.transform_to_pred(self.to_pred)
        
        # предсказание
        predicted = self.best_model.predict(self.to_pred)
        return predicted
        



class AutoRegressor():
    """
    Автоматический регрессор - не готов
    """
    def __init__(self):
        self.scores = {}
        self.models = []
        
    
    def transform_to_fit(self, X):
        pass
    
    
    def transform_to_pred(self, to_pred):
        pass
    
    
    def fit(self, X, y):
        pass
    
    
    def predict(self, to_pred):
        pass
    