#part 2: classification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random
import math

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict

#metrics
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score,classification_report,roc_auc_score

#function for Classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

train_filename = "adult.data"
test_filename = "adult.test"

#Listing of attributes:
#>50K, <=50K.

#age: continuous.
#workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
#fnlwgt: continuous.
#education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
#education-num: continuous.
#marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
#occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
#relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
#race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
#sex: Female, Male.
#capital-gain: continuous.
#capital-loss: continuous.
#hours-per-week: continuous.
#native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.


def load_data(train_filename,test_filename):
    feature_name = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss','hours-per-week', 'native-country',
                    'class']
    train_df = pd.read_csv(train_filename, sep=',', header=None,names=feature_name)
    test_df = pd.read_csv(test_filename, skiprows=[0], sep=',', header=None,names=feature_name)

    #replace missing value
    train_df = train_df.replace(" ?", value=np.NaN)
    test_df = test_df.replace(" ?", value=np.NaN)

    print("Missing values of training file:", train_df.isnull().values.any())
    print("Missing values of test file:", test_df.isnull().values.any())

    #replace class
    train_df["class"]=train_df["class"].replace(" >50K",1)
    test_df["class"] = test_df["class"].replace(" >50K.", 1)
    train_df["class"] = train_df["class"].replace(" <=50K", 0)
    test_df["class"] = test_df["class"].replace(" <=50K.", 0)

    #print data
    print('-----------data overview----------')
    print('size of data : ')
    print(train_df.shape)
    print(test_df.shape)
    print('features of data : ')
    print(train_df.columns)
    print(train_df.head())
    print(train_df.describe(include='all').to_string())
    print(test_df.describe(include='all').to_string())
    print("-----------data loaded------------")
    return train_df,test_df


def unlabeldata(data_set,label_name = "name"):
    data_data = data_set.copy();
    print(data_data.groupby('class').count())
    data_unlabel = data_data.drop([label_name],axis=1)
    data_labels = data_data[label_name]
    print("dropped label data:")
    print(data_unlabel.head())
    print("Label")
    return data_unlabel, data_labels

def classification_prediciton(test_label,prediction,technique="technique"):
    print('--------result of '+technique +'--------')
    print("Classification accuracy: "+ str(round(accuracy_score(test_label, prediction),2)))
    print("Classification precision: "+ str(round(precision_score(test_label, prediction),2)))
    print("Recall Score: "+str(round(recall_score(test_label, prediction),2)))
    print("f1 Score: "+str(round(f1_score(test_label, prediction),2)))
    print("AUC: "+str(round(roc_auc_score(test_label, prediction),2)))
    print("Classification report: \n "+ classification_report(test_label,prediction))
    print()
    print("-------------------------------------------")


if __name__ == '__main__':
    train_df,test_df = load_data(train_filename,test_filename)
    #replace missing value with most frequent attributes
    train_df = train_df.apply(lambda x: x.fillna(x.value_counts().index[0]))
    test_df = test_df.apply(lambda x: x.fillna(x.value_counts().index[0]))

    print("Missing values of training file after replacing:", train_df.isnull().values.any())
    print("Missing values of test file:", test_df.isnull().values.any())

    #unlabel dataset
    train_features,train_label=unlabeldata(train_df,'class')
    test_features,test_label=unlabeldata(test_df,'class')

    #one hot coding--dummies
    train_features = pd.get_dummies(train_features, columns=train_features.select_dtypes(include=['object']).columns)
    test_features = pd.get_dummies(test_features,columns=test_features.select_dtypes(include=['object']).columns)
    #drop attribute that not in test, otherwise different nums
    train_features = train_features.drop(["native-country_ Holand-Netherlands"], axis=1)

    #Classification process
    # function "KNN"
    print("KNN running...\n")
    start_time = datetime.datetime.now()
    KNN_classifer = KNeighborsClassifier()
    KNN_classifer.fit(train_features,train_label)
    KNN_testlabel_pred = KNN_classifer.predict(test_features)
    classification_prediciton(test_label,KNN_testlabel_pred,"KNN")
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    print("Execution time: " + str(execution_time))

    # function "naive Bayes"
    print("naive Bayes running...\n")
    start_time = datetime.datetime.now()
    NB_classifer = GaussianNB()
    NB_classifer.fit(train_features, train_label)
    NB_testlabel_pred = NB_classifer.predict(test_features)
    classification_prediciton(test_label, NB_testlabel_pred, "naive Bayes")
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    print("Execution time: " + str(execution_time))

    # function "SVM"
    print("SVM running...\n")
    start_time = datetime.datetime.now()
    SVM_classifer = svm.SVC(gamma='scale')
    SVM_classifer.fit(train_features, train_label)
    SVM_testlabel_pred = SVM_classifer.predict(test_features)
    classification_prediciton(test_label, SVM_testlabel_pred, "SVM")
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    print("Execution time: " + str(execution_time))

    # function " decision tree"
    print(" decision tree running...\n")
    start_time = datetime.datetime.now()
    DT_classifer = DecisionTreeClassifier()
    DT_classifer.fit(train_features, train_label)
    DT_testlabel_pred = DT_classifer.predict(test_features)
    classification_prediciton(test_label, DT_testlabel_pred, "decision tree")
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    print("Execution time: " + str(execution_time))

    # function " random forest"
    print(" random forest running...\n")
    start_time = datetime.datetime.now()
    RF_classifer = RandomForestClassifier()
    RF_classifer.fit(train_features, train_label)
    RF_testlabel_pred = RF_classifer.predict(test_features)
    classification_prediciton(test_label, RF_testlabel_pred, "random forest")
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    print("Execution time: " + str(execution_time))

    # function " AdaBoost"
    print(" AdaBoost running...\n")
    start_time = datetime.datetime.now()
    AB_classifer = AdaBoostClassifier()
    AB_classifer.fit(train_features, train_label)
    AB_testlabel_pred = AB_classifer.predict(test_features)
    classification_prediciton(test_label, AB_testlabel_pred, "AdaBoost")
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    print("Execution time: " + str(execution_time))

    # function " Gradient Boosting"
    print(" Gradient Boosting running...\n")
    start_time = datetime.datetime.now()
    GB_classifer = GradientBoostingClassifier()
    GB_classifer.fit(train_features, train_label)
    GB_testlabel_pred = GB_classifer.predict(test_features)
    classification_prediciton(test_label, GB_testlabel_pred, "Gradient Boosting")
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    print("Execution time: " + str(execution_time))

    # function " linear discriminant analysis"
    print(" linear discriminant analysis running...\n")
    start_time = datetime.datetime.now()
    LDA_classifer = LinearDiscriminantAnalysis()
    LDA_classifer.fit(train_features, train_label)
    LDA_testlabel_pred = LDA_classifer.predict(test_features)
    classification_prediciton(test_label, LDA_testlabel_pred, "linear discriminant analysis")
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    print("Execution time: " + str(execution_time))

    # function " multi-layer perceptron"
    print(" multi-layer perceptron running...\n")
    start_time = datetime.datetime.now()
    MLP_classifer = MLPClassifier()
    MLP_classifer.fit(train_features, train_label)
    MLP_testlabel_pred = MLP_classifer.predict(test_features)
    classification_prediciton(test_label, MLP_testlabel_pred, "multi-layer perceptron")
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    print("Execution time: " + str(execution_time))

    # function " logistic regression"
    print(" logistic regression running...\n")
    start_time = datetime.datetime.now()
    LogR_classifer = LogisticRegression()
    LogR_classifer.fit(train_features, train_label)
    LogR_testlabel_pred = LogR_classifer.predict(test_features)
    classification_prediciton(test_label, LogR_testlabel_pred, "logistic regression")
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    print("Execution time: " + str(execution_time))