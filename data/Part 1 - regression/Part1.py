import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import random
import math
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict

#metrics
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score,classification_report
from sklearn.metrics import scorer,mean_absolute_error,mean_squared_error,r2_score

#function for regression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor

seed = 309
np.random.seed(seed)
filename = "diamonds.csv"
train_test_split_size = 0.3


def load_data(filename):
    feature_name = ['index','carat','cut','color','clarity','depth','table','x','y','z','price']
    df = pd.read_csv(filename, sep=',')
    df = convert_data_type(df)
    #df.hist(bins=10, figsize=(14, 10))
    #sns.factorplot(data=df, kind='box', size=4, aspect=2.5)
    #df = df.cumsum()
    #plt.figure();
    #df.plot()
    #plt.show()
    print('-----------data overview----------')
    print(df.corr())
    print('size of data : ')
    print(df.shape)
    print('features of data : ')
    print(df.head())
    print(df.describe(include='all').to_string())
    print("-----------data loaded------------")
    print("Missing values:", df.isnull().values.any())
    #df = getdummies(df)
    return df


def sum_bar_plot(df,feature_name="name"):
    plt.clf()
    df.groupby(feature_name).sum().plot(kind='bar')
    plt.show()

def getdummies(df):
    return pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns)


def convert_data_type(dataset):
    print("---------convert string data to numeric in dataset---------")
    dataset['cut'] = dataset['cut'].map({'Ideal':1,'Premium':2,'Very Good':3,'Good':4,'Fair':5})
    dataset['color'] = dataset['color'].map({'D':1,'E':2,'F':3,'G':4,'H':5,'I':6,'J':7})
    dataset['clarity'] = dataset['clarity'].map({'IF':0,'VVS1':1,'VVS2':2,'VS1':3,'VS2':4,'SI1':5,'SI2':6,'I1':7})
    dataset = dataset.drop(['index'],axis=1)

    return dataset

def dataset_split(df):
    # datasplit
    train_data, test_data = train_test_split(df, test_size=train_test_split_size,random_state=seed)
    print('splited train')
    print(train_data.shape)
    print('splited test ')
    print(test_data.shape)
    return train_data,test_data

def unlabeldata(test_set):
    test_data = test_set.copy();
    test_unlabel = test_data.drop(["price"], axis=1)
    test_labels = test_data["price"]
    print("dropped label data:")
    print(test_unlabel.shape)
    print("Label")
    print(test_labels.shape)
    return test_unlabel, test_labels

def regression_prediciton(test_label,prediction,technique="technique"):
    print('--------result of '+technique +'--------')
    mse = mean_squared_error(test_label, prediction)
    mae = mean_absolute_error(test_label,prediction)
    rmse = math.sqrt(mse)
    r2=r2_score(test_label,prediction)
    print("Mean squared error: " + str(round(mse,2)))
    print("Root Mean squared error: "+str(round(rmse,2)))
    print("Mean absolute error: "+ str(round(mae,2)))
    print("r2 score: " + str(round(r2,2)))
    print("-------------------------------------------")

def drawscatter(df):
    size = 3
    f, ax = plt.subplots(3, 3, figsize=(150, 70))
    sns.scatterplot(x="carat", y="price", s=size, linewidth=0,
                    data=df, ax=ax[0, 0])
    sns.scatterplot(x="cut", y="price", s=size, linewidth=0,
                    data=df, ax=ax[0, 1])
    sns.scatterplot(x="color", y="price", s=size, linewidth=0,
                    data=df, ax=ax[0, 2])
    sns.scatterplot(x="clarity", y="price", s=size, linewidth=0,
                    data=df, ax=ax[1, 0])
    sns.scatterplot(x="depth", y="price", s=size, linewidth=0,
                    data=df, ax=ax[1, 1])
    sns.scatterplot(x="table", y="price", s=size, linewidth=0,
                    data=df, ax=ax[1, 2])
    sns.scatterplot(x="x", y="price", s=size, linewidth=0,
                    data=df, ax=ax[2, 0])
    sns.scatterplot(x="y", y="price", s=size, linewidth=0,
                    data=df, ax=ax[2, 1])
    sns.scatterplot(x="z", y="price", s=size, linewidth=0,
                    data=df, ax=ax[2, 2])

def drawBox(x_train):
    feature_name = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
    std_xtrain = pd.DataFrame(x_train)
    std_xtrain.columns = feature_name
    sns.factorplot(data=std_xtrain, kind='box', size=4, aspect=2.5)
    #std_xtrain.hist(bins=10, figsize=(14, 10))


def remove_outliers_IQR(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_out = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    df_out = df_out.dropna()
    return df_out

if __name__ == '__main__':
    df = load_data(filename)
    train_data,test_data=dataset_split(df)

    #remove outliers based on IQR
    #train_data = remove_outliers_IQR(train_data)

    #draw scatter plot for each feature
    #drawscatter(df)

    #feature names
    feature_name = [ 'carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']

    #unlabel dataset
    x_test,y_test =unlabeldata(test_data)
    x_train,y_train = unlabeldata(train_data)

    #Plot hist after replace missing values
    #df.hist(bins=10, figsize=(14, 10))

    #Box plot before standardlize
    #drawBox(x_train)

    #standardlize
    stdlier = StandardScaler()
    stdlier.fit(x_train)
    x_train = stdlier.transform(x_train)
    x_test = stdlier.transform(x_test)

    #x train boxplot after standardlize
    #drawBox(x_train)

    #x_train = remove_outliers_IQR(x_train)
    #drawBox(x_train)
    #plt.show()

    #function "linear regression"
    print("#########################")
    start_time = datetime.datetime.now()
    LR_classifer = LinearRegression(fit_intercept=True, normalize=False)
    #LR_y_test_pred = cross_val_predict(LR_classifer,x_train,y_train,cv=10)
    LR_classifer.fit(x_train,y_train)
    LR_y_test_pred = LR_classifer.predict(x_test)
    #LR_y_test_pred_score = LR_classifer._decision_function(x_test)
    #print(LR_classifer.score(x_test,y_test))
    regression_prediciton(y_test, LR_y_test_pred, "Linear regression")
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    print("Execution time: " + str(execution_time))

    # function " k-neighbors regression"
    print("#########################")
    start_time = datetime.datetime.now()
    KN_classifer = KNeighborsRegressor(n_neighbors=11,algorithm='auto',leaf_size=15, p=2)#n_neighbors=5,algorithm='auto'
    KN_classifer.fit(x_train, y_train)
    KN_y_test_pred = KN_classifer.predict(x_test)
    regression_prediciton(y_test, KN_y_test_pred, "k-neighbors regression")
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    print("Execution time: " + str(execution_time))

    # function "Ridge regression"
    print("#########################")
    start_time = datetime.datetime.now()
    RD_classifer = Ridge(alpha=0.5, fit_intercept=True, normalize=False)#alpha=1
    RD_classifer.fit(x_train, y_train)
    RD_y_test_pred = RD_classifer.predict(x_test)
    regression_prediciton(y_test, RD_y_test_pred, "Ridge regression")
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    print("Execution time: " + str(execution_time))

    # function " decision tree regression"
    print("#########################")
    start_time = datetime.datetime.now()
    DT_classifer = DecisionTreeRegressor(criterion='mse',random_state=30)#max_depth=2
    DT_classifer.fit(x_train, y_train)
    DT_y_test_pred = DT_classifer.predict(x_test)
    regression_prediciton(y_test, DT_y_test_pred, "decision tree regression")
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    print("Execution time: " + str(execution_time))

    # function "random forest regression"
    print("#########################")
    start_time = datetime.datetime.now()
    RF_classifer = RandomForestRegressor(n_estimators=300)#max_depth=2,random_state=0,n_estimators=100
    RF_classifer.fit(x_train, y_train)
    RF_y_test_pred = RF_classifer.predict(x_test)
    regression_prediciton(y_test, RF_y_test_pred, "random forest regression")
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    print("Execution time: " + str(execution_time))

    # function "gradient Boosting regression"
    print("#########################")
    start_time = datetime.datetime.now()
    GB_classifer = GradientBoostingRegressor(learning_rate=0.02,n_estimators = 1000,tol=1e-4)#tol=1e-4,learning_rate = 0.02,n_estimators = 1000, max_depth = 5,criterion = 'friedman_mse'
    GB_classifer.fit(x_train, y_train)
    GB_y_test_pred = GB_classifer.predict(x_test)
    regression_prediciton(y_test, GB_y_test_pred, "gradient Boosting regression")
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    print("Execution time: " + str(execution_time))

    # function "SGD regression"
    print("#########################")
    start_time = datetime.datetime.now()
    SGD_classifer = SGDRegressor(max_iter=1000, tol=1e-3,early_stopping=True)#max_iter=100, tol=1e-3
    SGD_classifer.fit(x_train, y_train)
    SGD_y_test_pred = SGD_classifer.predict(x_test)
    regression_prediciton(y_test, SGD_y_test_pred, "SGD regression")
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    print("Execution time: " + str(execution_time))

    # function "support vector regression"
    print("#########################")
    start_time = datetime.datetime.now()
    SVR_classifer = SVR(C=1000.0,kernel='linear')
    SVR_classifer.fit(x_train, y_train)
    SVR_y_test_pred = SVR_classifer.predict(x_test)
    regression_prediciton(y_test, SVR_y_test_pred, "support vector regression")
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    print("Execution time: " + str(execution_time))

    # function "Linear SVR"
    print("#########################")
    start_time = datetime.datetime.now()
    LSVR_classifer = LinearSVR(C = 10.0,loss = 'squared_epsilon_insensitive',dual = True)#C = 1e-3,loss = 'squared_epsilon_insensitive',dual = True
    LSVR_classifer.fit(x_train, y_train)
    LSVR_y_test_pred = LSVR_classifer.predict(x_test)
    regression_prediciton(y_test, LSVR_y_test_pred, "Linear SVR")
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    print("Execution time: " + str(execution_time))

    # function " multi-layer perceptron regression"
    print("#########################")
    start_time = datetime.datetime.now()
    MLP_classifer = MLPRegressor(learning_rate_init=0.005,early_stopping=True)
    MLP_classifer.fit(x_train, y_train)
    MLP_y_test_pred = MLP_classifer.predict(x_test)
    regression_prediciton(y_test, MLP_y_test_pred, "multi-layer perceptron regression")
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    print("Execution time: " + str(execution_time))
    print("----------------End process----------------")

