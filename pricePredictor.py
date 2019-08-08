def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_score
from sklearn import svm




#reuires anaconda
#reuiqres sklearn

def main():
    print("opening file...")
    inputFile = open("craigslistVehiclesFull.csv",encoding="utf8")
    print("reading data...")
    importantAttributes = ["city","price","year","manufacturer","make","condition","cylinders","odometer","title_status","transmission"]
    df = pd.read_csv(inputFile, usecols=importantAttributes) #read_csv, only selecting attributes with the most variance compared to price
    
    filtered = df[(df['city'] == 'losangeles')] #filter out losangeles data
    filtered = filtered.sample(frac=1).reset_index(drop=True)  #shuffle dataset
    filtered.dropna(inplace=True) #drop rows with empty columns
    
    #hard cast categorical columns for encoding
    filtered['manufacturer'] = filtered.manufacturer.astype(str)
    filtered['make'] = filtered.make.astype(str)
    filtered['condition'] = filtered.condition.astype(str)
    filtered['cylinders'] = filtered.cylinders.astype(str)
    filtered['transmission'] = filtered.transmission.astype(str)
    
    #encode categorical columns as integers
    print("Encoding values...")
    for column in filtered.columns:
        if filtered[column].dtype == type(object):
            le = preprocessing.LabelEncoder()
            filtered[column] = le.fit_transform(filtered[column])
    
    feature_cols = ["year","manufacturer","make","condition","cylinders","odometer","title_status","transmission"]
    
    X=filtered[feature_cols]
    y=filtered.price
        
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.20, random_state = 100) #split the dataset into training and test data
    
    dimitri = DecisionTreeClassifier() #classify a decision tree
    
    print("performing 10-fold cross validation...")
    model = svm.SVC()
    accuracy = cross_val_score(model, X, y, scoring='accuracy', cv = 10) #perform cross validation
    print("Accuracy of Model with Cross Validation is:",accuracy.mean() * 100)
    
    print("training decision tree...")
    dimitri = dimitri.fit(X_train,y_train) #train it on the training data
    
    print("making price predictions...")
    y_pred = dimitri.predict(X_test) #use the decision tree to make predictions on test data
    print("Accuracy on test dataset is ", accuracy_score(y_test,y_pred)*100)
    
    inputFile.close()
    
if __name__ == '__main__':
    main()