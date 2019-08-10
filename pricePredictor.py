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
    
    importantAttributes = ['city','price', 'year', 'manufacturer', 'condition', 'fuel', 'odometer', 'paint_color']
    df = pd.read_csv(inputFile, usecols=importantAttributes) #read_csv, only selecting attributes with the most variance compared to price
    
    LAdf = df[(df['city'] == 'losangeles')] #filter out losangeles data

    LAdf = LAdf.sample(frac=1).reset_index(drop=True)  #shuffle dataset
    
    #remove outliers from dataset
    Q1 = LAdf.quantile(0.30)
    Q3 = LAdf.quantile(0.80)
    IQR = Q3 - Q1
    filteredLAdf = LAdf[~((LAdf < (Q1 - 1.5 * IQR)) |(LAdf > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    #fill NaN values with the most commonly occuring value
    filteredFilledLAdf = filteredLAdf.fillna(filteredLAdf.mode().iloc[0])
    
    #encode categorical columns as integers
    print("Encoding values...")
    encoded = filteredFilledLAdf.apply(preprocessing.LabelEncoder().fit_transform)
    
    feature_cols = ['price', 'year', 'manufacturer', 'condition', 'fuel', 'odometer', 'paint_color']
    X=encoded[feature_cols]
    y=encoded.price
        
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.20, random_state = 100) #split the dataset into training and test data
    
    dimitri = DecisionTreeClassifier() #classify a decision tree
    
    print("training decision tree...")
    dimitri = dimitri.fit(X_train,y_train) #train it on the training data
    
    print("making price predictions...")
    y_pred = dimitri.predict(X_test) #use the decision tree to make predictions on test data
    print("Accuracy on test dataset is ", accuracy_score(y_test,y_pred)*100)
    
    print("performing 10-fold cross validation...")
    accuracy = cross_val_score(dimitri, encoded, y, scoring='f1_macro', cv = 10) #perform cross validation
    print("Accuracy of Model with Cross Validation is:",accuracy.mean() * 100)
    
    inputFile.close()
    
if __name__ == '__main__':
    main()