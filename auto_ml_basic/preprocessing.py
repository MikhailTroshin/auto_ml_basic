# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 23:55:34 2020

@author: Troshin Mikhail
m.troshin.ml@gmail.com

"""

# importing libraries
import pandas as pd
import numpy as np
import feature_selector
from itertools import chain
   
# ______________________________ MAIN CLASSES ____________________________

# class for preprocess numerical features
class NumPreprocessor:
    """
    Класс для подготовки числовых признаков
    """
    
    def __init__(self, feature_selection=True, anomaly_detection=False, standardization=True):
        
        self.feature_selection = feature_selection
        self.anomaly_detection = anomaly_detection
        self.standardization = standardization
        
        # Проверяем ввод аргументов
        if (type(self.standardization) != bool or 
            type(self.feature_selection) != bool or 
            type(self.anomaly_detection) != bool):
            raise TypeError('The type of all arguments has to be boolean!')   
        
    def _standardize_fit_transform(self):
        from sklearn.preprocessing import StandardScaler
        self._sc = StandardScaler()
        self.data = self._sc.fit_transform(self.data)
    
    def _standardize_transform(self):
        self.to_predict = self._sc.transform(self.to_predict)
    
    def _select_features(self):
        from feature_selector import FeatureSelector
        self._fs = FeatureSelector(data = self.data)
        self._fs.identify_missing(missing_threshold=0.5)
        self._fs.identify_collinear(correlation_threshold = 0.95)
        self.data = self._fs.remove(methods=['missing','collinear'])
        self._to_drop = [] # list of features (columns) to drop
        self._to_drop.append(self._fs.ops['missing'])
        self._to_drop.append(self._fs.ops['collinear'])
        # make a set of features to drop
        self._to_drop = set(list(chain(*self._to_drop)))
    
    def _detect_anomaly(self):
        print('The method is not ready yet. Input data has not been modified.')
        
    
    @staticmethod
    def fill_missing(dataset):
        missing = dataset.isnull().sum() # Series object
        for index_name in missing.index:
            if missing[index_name] > 0:
                dataset[index_name] = dataset[index_name].fillna(dataset[index_name].mean())
        return dataset
    
    def fit_transform(self, X):
        """
        Метод первичного преобразования данных для корректной работы модели.
        Последовательно выполняются:
            - отбор признаков
            - поиск аномалий
            - стандартизация значений

        Parameters
        ----------
        X : pd.DataFrame
            DESCRIPTION.

        Returns
        -------
        pd.DataFrame
            DESCRIPTION.

        """
        self.data = X
        _indices = X.index
        _columns = X.columns
        
        if self.feature_selection:
            self._select_features()

        self.data = self.fill_missing(self.data)
        
        if self.anomaly_detection:
            self._detect_anomaly()
            
        if self.standardization:
            self._standardize_fit_transform()
            
        self.data = pd.DataFrame(self.data, index=_indices, columns=_columns)
        return self.data
        
        
    def transform(self, to_predict):
        """
        Метод для трансформации празнаков при подготовке предсказания

        Parameters
        ----------
        to_predict : pd.DataFrame
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        pd.DataFrame
            DESCRIPTION.

        """
        self.to_predict = to_predict
        _pred_indices = to_predict.index
        _pred_columns = to_predict.columns
        
        if self.feature_selection: # использовали отбор признаков
            if not self._fs:
                raise Exception('Use ".fit_transform" method first')
            else:
                # удаление отобранных на обучении признаков
                self.to_predict = self.to_predict.drop(columns = self._to_drop)
                
        self.to_predict = self.fill_missing(self.to_predict)
                
        if self.anomaly_detection:
            pass
        
        if self.standardization:
            self._standardize_transform()
            
        self.to_predict = pd.DataFrame(self.to_predict, index=_pred_indices, columns=_pred_columns)
        return self.to_predict


# class for preprocess label or categorical features 
class LabelPreprocessor:
    """
    Класс для подготовки категориальных признаков и лейблов
    """
    def __init__(self, feature_selection=True):

        self.feature_selection = feature_selection
        
        # Проверяем ввод аргументов
        if type(self.feature_selection) != bool:
            raise TypeError('The type of all arguments has to be boolean!')
    
    def _select_features(self):
        from feature_selector import FeatureSelector
        self._fs = FeatureSelector(data = self.data)
        self._fs.identify_missing(missing_threshold=0.5)
        self.data = self._fs.remove(methods=['missing'])
        self._to_drop = [] # list of features (columns) to drop
        self._to_drop.append(self._fs.ops['missing'])
        # make a set of features to drop
        self._to_drop = set(list(chain(*self._to_drop)))
    
    @staticmethod
    def fill_missing(dataset):
        missing = dataset.isnull().sum() # Series object
        for index_name in missing.index:
            if missing[index_name] > 0:
                dataset[index_name] = dataset[index_name].fillna(dataset[index_name].mode()[0])
        return dataset
    
    def fit_transform(self, X):
        """
        Метод первичного преобразования данных для корректной работы модели.
        Последовательно выполняются:
            - отбор признаков
            - заполнение пропусков
            - получение dummy-переменных

        Parameters
        ----------
        X : pd.DataFrame
            DESCRIPTION.

        Returns
        -------
        pd.DataFrame
            DESCRIPTION.

        """
        self.data = X
        
        if self.feature_selection:
            self._select_features()
            
        self.data = self.fill_missing(self.data)
        self.data = pd.get_dummies(self.data, columns = self.data.columns)
        self._columns = self.data.columns #список фичей после преобразования
        return self.data
        
        
    def transform(self, to_predict):
        """
        Метод для трансформации празнаков при подготовке предсказания

        Parameters
        ----------
        to_predict : pd.DataFrame
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        pd.DataFrame
            DESCRIPTION.

        """
        self.to_predict = to_predict
        
        if self.feature_selection: # использовали отбор признаков
            if not self._fs:
                raise Exception('Use ".fit_transform" method first')
            else:
                # удаление отобранных на обучении признаков
                self.to_predict = self.to_predict.drop(columns = self._to_drop)
                
        self.to_predict = self.fill_missing(self.to_predict) 
        self.to_predict = pd.get_dummies(self.to_predict, columns = self.to_predict.columns)
        self._pred_cols = self.to_predict.columns
        # изменяем столбцы так, чтобы структура для предсказания совпадала со структурой на обучении
        _missing_cols = set(self._columns) - set(self._pred_cols)
        # Добавляем столбцы со значениями по умолчаниями = 0
        for col in _missing_cols:
            self.to_predict[col] = 0
        # вырявниваем порядок
        self.to_predict = self.to_predict[self._columns]
            
        return self.to_predict


# class for preprocess text features
class TextPreprocessor:
    def __init__(self, stemmer=None):
        """
        

        Parameters
        ----------
        stemmer : string, optional
            The default is None.
            - 'Porter' switches on Porter stemming algorithm
            - 'Snowball' switches on Snowball stemming algorithm

        Raises
        ------
        TypeError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.stemmer = stemmer
        # Проверяем ввод аргументов
        if type(self.stemmer) != str:
            raise TypeError('The type of "stemmer" argument has to be string!')
            
        
    
    @staticmethod
    def fill_missing(dataset):
        missing = dataset.isnull().sum() # Series object
        for index_name in missing.index:
            if missing[index_name] > 0:
                dataset[index_name] = dataset[index_name].fillna('None')
        return dataset
    
    
    @staticmethod
    def clean_text(text, stemmer = None):
        """
        Function cleans text for further 'Bag of words' or other NLP algorithms. 
        
        Parameters
        ----------
        text : string
            DESCRIPTION.
    
        Returns
        -------
        String
    
        """
        import re 
        # preprocess text
        result = re.sub('[^a-zA-Z]', ' ', text)
        result = result.lower()
        result = result.split()
        # stem if needed
        if stemmer == 'Porter':
            import nltk
            nltk.download('stopwords')
            from nltk.corpus import stopwords
            from nltk.stem.porter import PorterStemmer
            ps = PorterStemmer()
            result = [ps.stem(word) for word in result if not word in set(stopwords.words('english'))]
        elif stemmer == 'Snowball':
            import nltk
            nltk.download('stopwords')
            from nltk.corpus import stopwords
            from nltk.stem.snowball import SnowballStemmer
            ss = SnowballStemmer(language='english')
            result = [ss.stem(word) for word in result if not word in set(stopwords.words('english'))]
        elif stemmer != None:
            raise Exception('Argument "stemmer" has to be one of the following: None, "Porter", "Snowball"')
        result = ' '.join(result)
        return result
    
    
    def fit_transform(self, X):
        self.data = X
        
        self.data = self.fill_missing(self.data)
        corpus = []
        for i in range(len(self.data)):
            cleaned = self.clean_text(self.data, stemmer = self.stemmer)
            corpus.append(cleaned)
 
        # TF-idf vectorization
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer()
        self.data = self._tfidf.fit_transform(corpus).toarray()
        
        return self.data
        
    
    def transform(self, to_predict):
        self.to_predict = to_predict       
        self.to_predict = self.fill_missing(self.to_predict)
        self.to_predict = self._tfidf.transform(self.to_predict)    
        return self.to_predict


# Testing
if __name__ == '__main__':

    data = np.array([[1, 2.3, np.nan], [2, np.nan, 'One'], [np.nan, 6, 'Two']])
    testset = pd.DataFrame({'Number': [1, 2, np.nan],
                            'Value': [2.3, np.nan, 6],
                            'String': [np.nan, 'One', "Two"]})
    
    testset = fill_missing(testset)
    test_ops = {'One': [1,2],
                'Two': [3,4],
                'Three': [5,6]}
    
    test_drop = []
    test_drop.append(test_ops['One'])
    test_drop.append(test_ops['Three']) # list of lists  
    test_drop = set(list(chain(*test_drop)))
    
    numprep = NumPreprocessor()
    new_data = numprep.fit_transform(X.iloc[:, [1,2]])
    
    y_df = y_df.iloc[:5, :]
    pd.get_dummies(y_df, columns = ['shit'])
    
    text_set = X.iloc[:, [0,3]]
    lblprep = LabelPreprocessor(feature_selection=False)
    text_set = lblprep.fit_transform(text_set)
