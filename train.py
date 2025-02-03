import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder , LabelEncoder , MinMaxScaler , StandardScaler
from sklearn.model_selection import train_test_split , cross_val_score , GridSearchCV , RandomizedSearchCV , cross_validate
from sklearn.metrics import confusion_matrix , accuracy_score , classification_report , recall_score  , precision_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")


data= pd.read_csv(r"C:\Users\Yash\OneDrive\Desktop\Deployment\data\bank-full.csv" , sep=';')
print(data.head())


data = data.rename(columns={'y': 'Target'})

data.replace(['unknown','?'], np.nan, inplace=True)

columns_to_drop = ['poutcome', 'contact']
data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])


X = data.drop(['Target'] , axis=1)
y = data['Target']
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.25 , random_state=25)

mode = X_train[['job', 'education']].dropna().mode().iloc[0]
print(mode)

X_test['job'].fillna(mode , inplace = True)
X_train['job'].fillna(mode , inplace = True)
X_test['education'].fillna(mode , inplace = True)
X_train['education'].fillna(mode , inplace = True)

median = X_train['balance'].dropna().median()
print(median)
X_test['balance'].fillna(median, inplace = True)
X_train['balance'].fillna(median, inplace = True)

####

labelEncoder= LabelEncoder()
y_train = labelEncoder.fit_transform(y_train)
y_test = labelEncoder.transform(y_test)

categorical_columns = X.select_dtypes(include=['category', 'object']).columns
# Create and fit OneHotEncoder on the training data
ohe = OneHotEncoder(drop='first', sparse_output=True, handle_unknown='ignore')
# Fit and transform on the training data (will create sparse matrix)
X_train_cat_sparse = ohe.fit_transform(X_train[categorical_columns])
# Transform the test data using the same encoder (sparse matrix as well)
X_test_cat_sparse = ohe.transform(X_test[categorical_columns])
# Convert back to DataFrame for readability (converting sparse to dense format for visualization purposes)
# It's optional to do this step, as the model will work with sparse format
X_train_cat = pd.DataFrame.sparse.from_spmatrix(X_train_cat_sparse, columns=ohe.get_feature_names_out(categorical_columns), index=X_train.index)
X_test_cat = pd.DataFrame.sparse.from_spmatrix(X_test_cat_sparse, 
columns=ohe.get_feature_names_out(categorical_columns), index=X_test.index)
# Drop original categorical columns from X_train and X_test
X_train = X_train.drop(columns=categorical_columns)
X_test = X_test.drop(columns=categorical_columns)
# Concatenate the encoded categorical columns back (sparse format will be preserved)
X_train = pd.concat([X_train, X_train_cat], axis=1)
X_test = pd.concat([X_test, X_test_cat], axis=1)
#Features to scale
Numerical_columns = X.select_dtypes(include=['int']).columns
# Initialize MinMaxScaler
scaler = MinMaxScaler()
# Fit the scaler on training data and transform
X_train[Numerical_columns] = scaler.fit_transform(X_train[Numerical_columns])
# Transform the test data using the same scaler
X_test[Numerical_columns] = scaler.transform(X_test[Numerical_columns])

m6 =RandomForestClassifier(n_estimators=100, 
                            criterion='entropy', 
                            max_depth=None, 
                            min_samples_split=2, 
                            min_samples_leaf=1, 
                            max_features=None, 
                            bootstrap=True, 
                            oob_score=False, 
                            n_jobs=None, 
                            random_state=None, 
                            verbose=0, 
                            warm_start=False, 
                            class_weight=None)

m6.fit(X_train,y_train)
pred_m6 = m6.predict(X_test)
acc_m6 = accuracy_score(y_test,pred_m6)
report = classification_report(y_test, pred_m6)
print("Classification Report of RandomForestClassifier\n", report)


## Save the Model 
joblib.dump(m6 , filename="model.pkl")