# Scripting Exercices

Below is a data mining and scripting exercise. Note that we will use it to evaluate : 
 1. problem solving skills.
 1. machine learning skills; and,
 1. programming skills;
<br></br>

**To Do :**
1. Follow the instructions below while maintaining a presentable (clean) script. Ideally we ask that you 
make your script available on your own github.com account (Use the free version here : https://github.com/).  
1. Send us the link to your final commit before 9am the day of your interview
1. During the interview, we will ask that you present your work (preprocessing, model training, performance assessment, results & discussion).
We encourage you to present the results using either a **notebook** or a **README** file. At the very least, you should ensure
that your results are presentable.
<br></br>

**Remember :**
1. Make sure to **apply best practices** as you move through the examples. (data preprocessing, missing values, hyper parameter 
search, model evaluation, result visualisation, etc.)
1. **Make assumptions** where necessary, we are interested in your approach primarily.
1. **A good story is as important as an algorithm**. We expect you to be able to communicate and present your ideas, methodology 
and implementations. 

**Good Luck!** 
<br></br>
<br></br>

## Exercise 1 : Fraudulent Transactions (Classification)
The file **fraud_prep.csv** contains credit card transactions. 
1. Evaluate multiple classification algorithms to identify whether the transactions are fraudulent or not.
1. Compare the performance of each model & identify the best performing one.
1. Present how your model generalizes and performs on unseen data.
1. Make sure to present all steps taken

**BONUS Points :** Can you think of some **unsupervised** methods to accomplish this same task? If so, describe them (do not script them)
<br></br>
<br></br>

## Exercise 2.  Crime Dataset (Regression)
The Crime Dataset contains **128 socio-economic features** from the US 1990 Census. The target is the crime rate per community.

Ref. : https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names

Using the **crime_prep.csv** file :
1. Identify the variables that are the most highly correlated with the target
1. Apply either dimensionality reduction or feature selection on the dataset
1. Evaluate multiple regression algorithms to predict the price of houses.
1. Compare the performance of each model & identify the best performing one.
1. Present how your model generalizes and performs on unseen data.
<br></br>
<br></br>

Question 1

Approach
Clean and Transform Data
Classification Algorithms
(1) Logit
(2) Desicion Tree
(3) Support Vector Machine
Train and validate Models
Out of Sample Comparison
(1) False Negative Metric
Main Conclusion
We have very small number of frauds => easy to overfit
Tree outperfroms in out-of-sample:
(1) simple logit
(2) SVM
Tuning of SVM and tree is hard and tricky
Ensembling approach might be the best, i.e. random forest


ata_folder = r'DataFolder'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn import linear_model
from sklearn import svm

%matplotlib inline


df_fraud = pd.read_csv(Data_folder + 'fraud_prep.csv')


# Select label (for supervised learning)
df_fraud.Class.value_counts()

0    284315
1       492
Name: Class, dtype: int64


print( "% of frauds is " + str( np.round( 100.*df_fraud.Class.value_counts()[1] / len(df_fraud), 2 ) ) )
print( "% of frauds is " + str( np.round( 100.*df_fraud.Class.value_counts()[1] / len(df_fraud), 2 ) ) )


Pre data cleaning:
Demean and standardized data

df_features = df_fraud.iloc[:,:-1].copy(deep = True)
mean = df_features.mean(axis = 0)
std = df_features.std(axis = 0)
df_features = df_features.sub (mean, axis = 1)
df_features = df_features.div(std, axis = 1)
Drop string variables (nan are everywhere)

df_features = df_features.dropna(axis = 1)
(1) Multiple Classification algo

X = df_features.copy(deep = True)
Y = df_fraud.Class
Divide data into training, cross validation and out of sample

def train_validate_test_split(df, train_percent=.4, validate_percent=.2, seed=1):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.ix[perm[:train_end]]
    validate = df.ix[perm[train_end:validate_end]]
    test = df.ix[perm[validate_end:]]
    return train, validate, test

X_train, X_validate, X_test = train_validate_test_split(df_features)
​
# Find indexes for Y
Y_train = Y.copy(deep = True)
Y_validate = Y.copy(deep = True)
Y_test = Y.copy(deep = True)
​
Y_train = Y_train[X_train.index]
Y_validate = Y_validate[X_validate.index]
Y_test = Y_test[X_test.index]

print("All three sets are representative")
print( "% of frauds in train is " + str( np.round( 100.*Y_train.value_counts()[1] / len(Y_train), 2 ) ) )
print( "% of frauds in validation is " + str( np.round( 100.*Y_validate.value_counts()[1] / len(Y_validate), 2 ) ) )
print( "% of frauds in test is " + str( np.round( 100.*Y_test.value_counts()[1] / len(Y_test), 2 ) ) )
All three sets are representative
% of frauds in train is 0.16
% of frauds in validation is 0.18
% of frauds in test is 0.18
Train all model on train set
(a) Logit
Logit requires an arbitrary chosen threhsold for classification
Logit does not work well with False negative due to small number of frauds
Logit works perfectly with False positive

logit = linear_model.LogisticRegression(C=1e5)
logit.fit(X_train, Y_train)
in_sample_fit = logit.predict(X_train )

# Transform logit into classifier: if prob > tau then fraud
threshold = 0.5
decision = 1* (in_sample_fit > threshold)
​
decision_df = pd.DataFrame(decision, columns={'Model'})
Data_df = pd.DataFrame( Y_train.values, columns = {"Data"})
Probability_df = decision_df.join(Data_df) # Column 0 = model, Column 1 = data
Main metrics of interests:

# When we have fraud in the data
tmp_df = Probability_df.copy(deep = True)
tmp_df = tmp_df.loc[ tmp_df.iloc[:,1] >0,:]   # data shows fraud
Prob = 1.*(tmp_df.iloc[:,0] == 0).sum()/ len(tmp_df)
print("Logit: False Negative " + str(Prob) )
​
# When we do not have fraud in the data
tmp_df = Probability_df.copy(deep = True)
tmp_df = tmp_df.loc[ tmp_df.iloc[:,1] ==0,:]   # data shows no fraud
Prob = 1.*(tmp_df.iloc[:,0] == 1).sum()/ len(tmp_df)
print("Logit: False Positive " + str(np.round(Prob,3)) )
Logit: False Negative 0.413978494624
Logit: False Positive 0.0
(b) Tree
Tree does not require threhsold to make a decision
Tree works perfectly, but it can be due to overfitting (see it later on out of sample data)

tree_model = tree.DecisionTreeClassifier()
tree_model.fit(X_train, Y_train)
decision = tree_model.predict(X_train)

decision_df = pd.DataFrame(decision, columns={'Model'})
Data_df = pd.DataFrame( Y_train.values, columns = {"Data"})
Probability_df = decision_df.join(Data_df) # Column 0 = model, Column 1 = data
Main metrics of interests:

# When we have fraud in the data
tmp_df = Probability_df.copy(deep = True)
tmp_df = tmp_df.loc[ tmp_df.iloc[:,1] >0,:]   # data shows fraud
Prob = 1.*(tmp_df.iloc[:,0] == 0).sum()/ len(tmp_df)
print("Tree: False Negative " + str(Prob) )
​
# When we do not have fraud in the data
tmp_df = Probability_df.copy(deep = True)
tmp_df = tmp_df.loc[ tmp_df.iloc[:,1] ==0,:]   # data shows no fraud
Prob = 1.*(tmp_df.iloc[:,0] == 1).sum()/ len(tmp_df)
print("Tree: False Positive " + str(np.round(Prob,3)) )
Tree: False Negative 0.0
Tree: False Positive 0.0
(c) SVM
SVM does not require threhsold to make a decision
SVM has problem with False negative; it is better than logit but worse than tree

svm_model = svm.SVC()
svm_model.fit(X_train, Y_train)
decision = svm_model.predict(X_train)

decision_df = pd.DataFrame(decision, columns={'Model'})
Data_df = pd.DataFrame( Y_train.values, columns = {"Data"})
Probability_df = decision_df.join(Data_df) # Column 0 = model, Column 1 = data
Main metrics of interests:

# When we have fraud in the data
tmp_df = Probability_df.copy(deep = True)
tmp_df = tmp_df.loc[ tmp_df.iloc[:,1] >0,:]   # data shows fraud
Prob = 1.*(tmp_df.iloc[:,0] == 0).sum()/ len(tmp_df)
print("SVM: False Negative " + str(np.round(Prob,3) ))
​
# When we do not have fraud in the data
tmp_df = Probability_df.copy(deep = True)
tmp_df = tmp_df.loc[ tmp_df.iloc[:,1] ==0,:]   # data shows no fraud
Prob = 1.*(tmp_df.iloc[:,0] == 1).sum()/ len(tmp_df)
print("SVM: False Positive " + str(np.round(Prob,3)) )
SVM: False Negative 0.194
SVM: False Positive 0.0
(2) - (3)
Out of Sample
Logit - out of sample
Results are consistent with insample training

X_joint = X_train.append( X_validate) 
Y_joint = pd.DataFrame(np.hstack((Y_train.values, Y_validate.values)))

logit = linear_model.LogisticRegression(C=1e5)
logit.fit(X_joint, Y_joint)
in_sample_fit = logit.predict(X_test )

# Transform logit into classifier: if prob > tau then fraud
threshold = 0.5
decision = 1* (in_sample_fit > threshold)
​
decision_df = pd.DataFrame(decision, columns={'Model'})
Data_df = pd.DataFrame( Y_test.values, columns = {"Data"})
Probability_df = decision_df.join(Data_df) # Column 0 = model, Column 1 = data

# When we have fraud in the data
tmp_df = Probability_df.copy(deep = True)
tmp_df = tmp_df.loc[ tmp_df.iloc[:,1] >0,:]   # data shows fraud
Prob = 1.*(tmp_df.iloc[:,0] == 0).sum()/ len(tmp_df)
print("Logit: False Negative " + str(Prob) )
​
# When we do not have fraud in the data
tmp_df = Probability_df.copy(deep = True)
tmp_df = tmp_df.loc[ tmp_df.iloc[:,1] ==0,:]   # data shows no fraud
Prob = 1.*(tmp_df.iloc[:,0] == 1).sum()/ len(tmp_df)
print("Logit: False Positive " + str(np.round(Prob,3)) )
Logit: False Negative 0.381188118812
Logit: False Positive 0.0
Tune Tree
It requires a lot of running time
Now, results are less impressive but still better than logit = > can work on more tuning (different parameters: depth of the tree, number of samples at internal nodes, maximum number of features)

min_samples_leaf_loop = [6,2,3] # Change number of samples at the external node
r_squared_loop = np.zeros(len(min_samples_leaf_loop))
​
for i in range(0, len(min_samples_leaf_loop)): # Loop over C
    print(i)
    #Train for fied C
    tree_model = tree.DecisionTreeClassifier(min_samples_leaf = min_samples_leaf_loop[i])
    tree_model.fit(X_train, Y_train)
    
    # Save loss function
    r_squared_loop[i] = tree_model.score(X_validate, Y_validate)
​
    
idx = np.argmax(r_squared_loop)
min_samples_leaf_optimal_tree = min_samples_leaf_loop[idx] 
​
0
1
2

# # Final result
tree_model = tree.DecisionTreeClassifier(min_samples_leaf = min_samples_leaf_optimal_tree)
tree_model.fit(X_train, Y_train)
decision = tree_model.predict(X_test)
​
decision_df = pd.DataFrame(decision, columns={'Model'})
Data_df = pd.DataFrame( Y_test.values, columns = {"Data"})
Probability_df = decision_df.join(Data_df) # Column 0 = model, Column 1 = data

# When we have fraud in the data
tmp_df = Probability_df.copy(deep = True)
tmp_df = tmp_df.loc[ tmp_df.iloc[:,1] >0,:]   # data shows fraud
Prob = 1.*(tmp_df.iloc[:,0] == 0).sum()/ len(tmp_df)
print("Tree: False Negative " + str(Prob) )
​
# When we do not have fraud in the data
tmp_df = Probability_df.copy(deep = True)
tmp_df = tmp_df.loc[ tmp_df.iloc[:,1] ==0,:]   # data shows no fraud
Prob = 1.*(tmp_df.iloc[:,0] == 1).sum()/ len(tmp_df)
print("Tree: False Positive " + str(np.round(Prob,3)) )
Tree: False Negative 0.252475247525
Tree: False Positive 0.0
Tune SVM
It requires a lot of running time (the most time consuming process)
Results are less impressive than in training sample
Does not beat Tree, but is better than logit

C_loop = [1,2] # Change penalty
r_squared_loop = np.zeros(len(C_loop))
​
for i in range(0, len(C_loop)): # Loop over C
    print(i)
    # Train for fied C
    svm_model = svm.SVC(C = C_loop[i])
    svm_model.fit(X_train, Y_train)
    in_sample_fit = svm_model.predict(X_train)
    
    # Save loss function
    r_squared_loop[i] = svm_model.score(X_validate, Y_validate)
​
    
idx = np.argmax(r_squared_loop)
C_optimal_SVM = C_loop[idx]    
0
1

# # Final result
svm_model = svm.SVC(C = C_optimal_SVM)
svm_model.fit(X_train, Y_train)
decision = svm_model.predict(X_test)
​
decision_df = pd.DataFrame(decision, columns={'Model'})
Data_df = pd.DataFrame( Y_test.values, columns = {"Data"})
Probability_df = decision_df.join(Data_df) # Column 0 = model, Column 1 = data

# When we have fraud in the data
tmp_df = Probability_df.copy(deep = True)
tmp_df = tmp_df.loc[ tmp_df.iloc[:,1] >0,:]   # data shows fraud
Prob = 1.*(tmp_df.iloc[:,0] == 0).sum()/ len(tmp_df)
print("Tree: False Negative " + str(Prob) )
​
# When we do not have fraud in the data
tmp_df = Probability_df.copy(deep = True)
tmp_df = tmp_df.loc[ tmp_df.iloc[:,1] ==0,:]   # data shows no fraud
Prob = 1.*(tmp_df.iloc[:,0] == 1).sum()/ len(tmp_df)
print("Tree: False Positive " + str(np.round(Prob,3)) )
Tree: False Negative 0.331683168317
Tree: False Positive 0.0
(4) Bonus:
We should use mixture of models to identify frauduent transaction:
a) Use PCA to identify factor structure if it exists

b) Fit mixture for these PCA (otherwise dimension is too high for mixture model)

Alternatively, clustering approach (i.e. k-nearest neighbors)
% of frauds is 0.17
