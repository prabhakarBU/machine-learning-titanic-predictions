import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import seaborn as sns

titanic_df = pd.read_csv('huge_1M_titanic.csv')

# Key Points to Explore:
# Columns: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked.
# What are we trying to predict? (Survived - whether the passenger survived or not).
# What features may be useful?

# print(titanic_df.head())
# print(titanic_df.info())
# print(titanic_df.isnull().sum())

# Chapter 2
# Tasks:
# Handle missing values in Age, Embarked, and drop Cabin due to too many missing values.
# Convert categorical columns like Sex and Embarked into numerical format using one-hot encoding.
# Normalize and scale features if needed.

#convert and fill all None ages to median
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)

# just a FUN check in between
# print(titanic_df[titanic_df['Embarked'].isnull() & titanic_df['Cabin'].isnull()])

# print(titanic_df[titanic_df['Embarked'].isnull()])
# print(titanic_df['Embarked'].mode()[0])
titanic_df['Embarked'] = titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0])


# drop cabin since too many Nans and probably not useful feature
titanic_df.drop('Cabin',axis=1,inplace=True)
# print(titanic_df.isnull().sum())

# Encoding Categorical Variables
# Why: Many machine learning algorithms require input data to be numeric. 
# Categorical variables (like 'Sex' or 'Embarked' in the Titanic dataset) must be converted to numbers.

# convert categorical column like 'Sex' , 'Embarked' in object/string format to 0s and 1s
# print(titanic_df[titanic_df['Sex'] == 'male']['Sex'])
# titanic_df.loc[titanic_df['Sex'] == 'male','Sex'] = 0
# titanic_df.loc[titanic_df['Sex'] == 'female','Sex'] = 1
# print('SEX converted to 0/1')
# print(titanic_df[titanic_df['Sex'] == 0]['Sex'].count())
# print(titanic_df[titanic_df['Sex'] == 1]['Sex'].count())
# print(titanic_df.head(15))

# One-Hot Encoding: Converts categorical variables into a series of binary columns.
# using one-hot encoding to achieve the same and dropping extra column
titanic_df = pd.get_dummies(titanic_df, columns=['Sex','Embarked'], drop_first=True)

# Label Encoding: Assigns each category a unique integer.
# from sklearn.preprocessing import LabelEncoder
# encoder = LabelEncoder()
# X['Sex'] = encoder.fit_transform(X['Sex'])

# doubts 
# print(titanic_df['Cabin'].dtype) # convert to string?
# print(titanic_df["Survived"].value_counts())
# print(titanic_df.nunique())

# print(titanic_df.describe())
# print(titanic_df['Age'].max())
# print(titanic_df['Age'].quantile(0.25))

# Chapter 3
# Feature Selection
# Why: Not all features are equally important. Removing irrelevant or redundant features can improve model performance and reduce overfitting.
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Distribution of passengers by class and survival
# sns.countplot(x='Pclass', hue='Survived', data=titanic_df)
# plt.title('Passenger Class vs Survival')
# plt.show()

# # Distribution of passengers by sex and survival
# sns.countplot(x='Sex_male', hue='Survived', data=titanic_df)
# plt.title('Gender vs Survival')
# plt.show()

# # Visualize age distribution
# sns.histplot(titanic_df['Age'], bins=30, kde=True)
# plt.title('Age Distribution of Passengers')
# plt.show()

# # Correlation heatmap
# plt.figure(figsize=(10, 6))
# sns.heatmap(titanic_df.corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Heatmap')
# plt.show()

# Correlation Matrix: Identify highly correlated features that might be redundant.
# 
# import seaborn as sns
# import matplotlib.pyplot as plt

# corr_matrix = X.corr()
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.show()

# Recursive Feature Elimination (RFE): Use an ML model (e.g., Decision Tree) to select the most important features.
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression

# rfe = RFE(estimator=LogisticRegression(), n_features_to_select=5)
# rfe.fit(X, y)
# selected_features = X.columns[rfe.support_]

# L1 Regularization: In models like Lasso Regression, features with less importance may have their coefficients shrink to zero, effectively being removed from the model.
# from sklearn.linear_model import Lasso

# lasso = Lasso(alpha=0.01)
# lasso.fit(X, y)
# important_features = X.columns[lasso.coef_ != 0]


# Chapter 4: Feature Engineering
# In this chapter, we’ll:

# Identify the most important features.
# Create new features if necessary (e.g., family size, fare per person).
# Scale numerical features if needed.
# Question: "What additional features can you engineer to help the model make better predictions? 
# For example, how could SibSp and Parch be combined to create a more informative feature?"

# Hint: You could create a new feature FamilySize by summing SibSp and Parch. 
# How do you think family size could impact survival rates on the Titanic?

titanic_df['FamilySize'] = titanic_df['SibSp'] + titanic_df['Parch']
titanic_df.drop(['Name','Ticket','PassengerId'],axis=1,inplace=True)
titanic_df['IsAlone'] = (titanic_df['FamilySize'] == 1).astype(int)
# print(titanic_df.head(15))

# Tasks:
# Create a new feature FamilySize.
# Drop irrelevant columns like Name, Ticket, and PassengerId.

## Chapter 5: Building Machine Learning Models
# In this chapter, we’ll:
# Split the data into training and testing sets.
# Train Linear Regression, Logistic Regression, and Decision Tree models.
# Evaluate the models using accuracy, precision, recall, and confusion matrix.

## Handling Imbalanced Data
# Why: In some datasets, one class may dominate the others
# (e.g., 90% of Titanic passengers survived). This imbalance can lead to biased models.
# Resampling: Balance the dataset by oversampling the minority class or undersampling the majority class.
from sklearn.utils import resample
# checking how balanced/distributed is the target column data
print(titanic_df['Survived'].value_counts())
# sns.countplot(x='Survived', data=titanic_df)
# Upsample minority class
df_minority = titanic_df[titanic_df['Survived'] == 1]
df_majority = titanic_df[titanic_df['Survived'] == 0]
df_minority_upsampled = resample(df_minority,replace=True,n_samples=len(df_majority),random_state=42)
titanic_df_rebalanced = pd.concat([df_majority, df_minority_upsampled])
print(titanic_df_rebalanced['Survived'].value_counts())

# X = titanic_df[['Pclass','Age','Sex_male']]
X = titanic_df_rebalanced.drop(['Survived'],axis=1)
y = titanic_df_rebalanced['Survived']
# print(X.head())
# print(y.head())

# Split into Train and Test data
from sklearn.model_selection import train_test_split

X_train,X_test,  y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Scaling and Normalization ( for Features )
# Why: Different features can have different scales. 
# For example, 'Age' might range from 0 to 80, while 'Fare' could range from 0 to 500. Many models, 
# like Logistic Regression, KNN, and SVM, assume that all features are on a similar scale.
from sklearn.preprocessing import StandardScaler
# Standardization (Z-score normalization): Scales data so that each feature has a mean of 0 and a standard deviation of 1.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# from sklearn.preprocessing import MinMaxScaler
# Scales data to a fixed range, usually [0, 1]
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)


# SMOTE (Synthetic Minority Over-sampling Technique): Generates synthetic samples for the minority class.
# from imblearn.over_sampling import SMOTE
# smote = SMOTE()
# X_resampled, y_resampled = smote.fit_resample(X, y)


# Option 3: Using class_weight in Models
# If you don’t want to resample the data, you can modify your models to account for class imbalance. 
# For example, both Logistic Regression and Decision Tree classifiers in sklearn have a class_weight parameter, which you can set to balanced.
# This will automatically adjust the weights inversely proportional to the class frequencies.
# Using class_weight='balanced':
# Logistic Regression with class weights
# log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
# log_model.fit(X_train_scaled, y_train)

# # Decision Tree with class weights
# tree_model = DecisionTreeClassifier(class_weight='balanced')
# tree_model.fit(X_train_scaled, y_train)


# Chapter 5: Building Machine Learning Models
# In this chapter, we’ll:

# Split the data into training and testing sets.
# Train Linear Regression, Logistic Regression, and Decision Tree models.
# Evaluate the models using accuracy, precision, recall, and confusion matrix.
from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X_train,y_train)

y_pred_log = logistic_model.predict(X_test_scaled)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))

# Summary:
# Detect imbalance in the target variable before training the model using value_counts() and visualizations like countplot.
# Handle the imbalance using techniques like resampling (oversampling or undersampling) or using class_weight in models like Logistic Regression or Decision Trees.
# Optionally, use advanced techniques like SMOTE to generate synthetic examples of the minority class.

# Saving and Deploying to local machine and loading using joblib
import joblib
# joblib.dump(logistic_model, 'titanic_logistic_regression_model.pkl')
# Load the model
loaded_model = joblib.load('titanic_logistic_regression_model.pkl')
# Predict with the loaded model
new_predictions = loaded_model.predict(X_test_scaled)

print("Logistic Regression Accuracy:", accuracy_score(y_test, new_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, new_predictions))
print("Classification Report:\n", classification_report(y_test, new_predictions))


# Saving and Deploying to local machine and loading using pickle
# Save the model
# import pickle
# with open('logistic_regression_titanic.pkl', 'wb') as file:
#     pickle.dump(logistic_model, file)

# Load the model
# with open('logistic_regression_titanic.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)

# Predict with the loaded model
# new_predictions = loaded_model.predict(X_test_scaled)
