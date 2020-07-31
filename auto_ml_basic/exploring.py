# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 23:54:55 2020

@author: Troshin Mikhail
m.troshin.ml@gmail.com

Module has built-in functions for following procedures:
    - feature selection (deleting correlated features)
    - anomaly detection (?)
    - detecting types of featrures and create feature-type categories - done
    - checking a type of task - done
 
"""

import pandas as pd
import numpy as np

numeric_types = ['bool', 'int64', 'float64']
str_types = ['object', 'string']
   

# detecting a type of independent variable
def if_for_regr(y, n_classes = 10):    
    """
    Функция исследует вектор ответов на входе и выдает True,
    если данные являются предсказаниями в задаче регресии,
    False, если они являются предсказаниями в задаче классификации

    Parameters
    ----------
    y : pd.Series
        DESCRIPTION.
    n_classes : int, optional
        DESCRIPTION. The default is 10.

    Returns: Bool
    -------
    None.

    """
    
    # Проверяем ввод аргументов
    if type(y) != pd.Series:
        raise TypeError('The type of input data has to be a pandas.Series!')   
    if type(n_classes) != int:
        raise TypeError('The type of "n_classes" variable has to be int!')      
    if n_classes < 2:
        raise Exception('The value of "n_classes" variable is less than 2')
        
    # Сравниваем с количеством уникальных значений   
    return True if (len(np.unique(y)) > n_classes) else False
    

# Feature categorization
def make_feature_categories(X, n_labels=30, n_str = 30):
    """
    Функция относит каждый из признаков (столбцов) к одному из 4-х классов:
    числовой, числовой идентификатор (лейбл), категориальный, текстовый.

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    n_labels : TYPE, optional
        DESCRIPTION. The default is 30.
    n_str : TYPE, optional
        DESCRIPTION. The default is 30.

    Returns
    -------
    categories : dict
        Словарь с 4-мя ключами-категориями и именами столбцов анализируемого DataFrame,
        отнесенными к соответствующей категории

    """
    
    # Проверяем ввод аргументов
    if type(X) != pd.DataFrame:
        raise TypeError('The type of input data has to be a pandas.DataFrame!')
    if len(X.shape) > 2:
        raise Exception('The input data has to be 2-dimentional')
    if type(n_labels) != int:
        raise TypeError('The type of "n_labels" variable has to be int!')      
    if n_labels < 2:
        raise Exception('The value of "n_labels" variable is less than 2')
    if type(n_str) != int:
        raise TypeError('The type of "n_labels" variable has to be int!')      
    if n_str < 2:
        raise Exception('The value of "n_str" variable is less than 2')
        
    categories = {'num': [], 
                  'cat': [], 
                  'labels': [], 
                  'text': []}
    
    for column in X.columns:
        # numerical categories
        if str(X[column].dtype) in numeric_types:
            if len(np.unique(X[column])) < n_labels:
                categories['labels'].append(column)
            else:
                categories['num'].append(column)
                
        # string categories
        elif str(X[column].dtype) in str_types:
            if len(np.unique(X[column])) < n_str:
                categories['cat'].append(column)
            else:
                categories['text'].append(column)
                
        # other categories
        else:
            raise Exception('The data type of the %s column is not supported' %str(column))
            
    return categories



# Testing
if __name__ == '__main__':
    
    dataset = pd.read_csv('Data.csv')
    #dataset = dataset[:100]
    dataset_dtypes_cats = make_feature_categories(dataset, n_labels=6, n_str=6)
        
        
    