#вклучување на потребните библиотеки
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing, cross_validation
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics.classification import accuracy_score
from sklearn.neighbors.classification import KNeighborsClassifier
from cmath import sqrt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier

def main():
    #вчитување на тренирачкото и тестирачкото множество и парсирање на датите во колоната Dates
    train = pd.read_csv('E:/train.csv', parse_dates=['Dates'])
    test = pd.read_csv('E:/test.csv', parse_dates=['Dates'])
    
    #трансформирање на секоја можна вредност на класниот атрибут Category во соодветна бројка
    le_crime = preprocessing.LabelEncoder()
    crime = le_crime.fit_transform(train.Category)
    
    #градење на data frame, каде име на секоја колона е можна вредност на атрибутот,
    #соодветно векторизирање на секој од атрибутите што се вклучени во процесот на градење на моделот
    #пример:
        #DayOfWeek -> M T W T F S S
        #Saturday -> [0 0 0 0 0 1 0] 
    days = pd.get_dummies(train.DayOfWeek)
    district = pd.get_dummies(train.PdDistrict)
    hour = train.Dates.dt.hour
    hour = pd.get_dummies(hour)
    
    #конкатенирање на векторизираните атрибути и додавање на класниот атрибут crime, кој сега е нумерички атрибут
    train_data = pd.concat([hour, days,district], axis=1)
    train_data['crime'] = crime
    
    #повторување на истата постапка само сега врз тестирачкото множество
    days = pd.get_dummies(test.DayOfWeek)
    district = pd.get_dummies(test.PdDistrict)
    hour = test.Dates.dt.hour
    hour = pd.get_dummies(hour)  
    
    test_data = pd.concat([hour, days, district], axis = 1)
    
    #во features ги додаваме сите колони на конкатенираниот data frame train_data, 
    #тоа се имиња на колони кои ќе ги користиме при градењето на моделот
    features = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday','Wednesday', 
                'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',
                'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN'] + [x for x in range(0, 24)]
    
    #поделба на тренирачкото множество на 2 дела:
        #70% - тренирачки дел
        #30% - валидациски дел
    training, validation = train_test_split(train_data, train_size=.70)
    
   
    #градење на Бајесовиот класификатор со тренирачкиот дел
    
   
    model = BernoulliNB()
    model.fit(training[features], training['crime'])
    
    
    
    
    #пресметување на log loss функцијата со валидацискиот дел и печатење на вредноста
    predicted = np.array(model.predict_proba(validation[features]))
    
    print(("Log loss ")+str(log_loss(validation['crime'], predicted)))
    
    
    
    #градење на моделот со целокупното тренирачко множество, 
    #предвидување на веројатностите за класата на тестирачкото множество и 
    #зачувување на добиените податоци и подготовка за поставување на Kaggle
    model = BernoulliNB()
    model.fit(train_data[features], train_data['crime'])
    predicted = model.predict_proba(test_data[features])
    result = pd.DataFrame(predicted, columns = le_crime.classes_)
    result.to_csv('submit.csv',index=True, index_label='Id')
    
    
    #предвидување на класата на секоја инстанца од тестирачкото множество,
    #инверзна трансформација на нумеричката вредност за да ја добиеме првобитната текстуална вредност за категоријата,
    #зачувување на добиените податоци во нов .csv документ
    predicted_val = model.predict(test_data[features])
    predicted_val = le_crime.inverse_transform(predicted_val)
    result = pd.DataFrame(predicted_val)
    result.to_csv('results.csv', index=False)
    
    
if __name__ == '__main__':
    main()