# Identifying-Potential-Customers

The objective of this project is to identify the potential customers that are likely to set up a term deposit at a Portuguese 
banking institution along with the key characteristics that add value to the marketing campaign. After 
preprocessing the data, that includes treating the imbalanced class through sampling techniques, we have 
evaluated the performance of several classification models such as Logistic regression, Decision Trees, Random 
Forests, Gradient Boosting Classifier, K-NN and SVM. Random Forest ended up as the best performing model
among all of them, where F1-Score has been chosen as the evaluation criteria.


The repository contains several jupyter notebooks and a rmd file that contains the initial data preprocessing and exploratory data analysis.


## Summary

- ### Background
Banks use depositor’s money to make loans. Term deposits which are deposits in a financial institution with a specific 
maturity rate are one of the major sources for a bank which they can use to make loans[1]. Banks follow various marketing 
strategies like email marketing, SMS/Phone call marketing, advertisements etc. to reach out to the customers. Phone call 
campaigning is one of the traditional forms of marketing and, when done suitably, can have the best results. Most of the 
businesses follow a priority queue where they shortlist the customers, they believe are likely to convert. Organizations allot 
a huge number of resources towards organizing such campaigns which makes the task of identifying potential customers a 
crucial one. Our aim is to identify such potential customers that the bank could target and help banks make optimal usage 
of their resources .

- ### Dataset
The data contains information about a marketing campaign conducted by the Central Bank of the Portuguese Republic[4]
with phone calls as the medium of communication. There are about 41,188 observations and 20 
variables/features including consumer price index, marital status, employment variation rate, average yearly balance etc[5]. 
The dataset is split into train and test sets in the ratio 80:20.

- ### Goal of the Project 
The main goal of the project is to identify the potential customers who are likely to set up a term deposit using a robust 
classifier based on relevant features/predictors. We intend to identify the key characteristics that makes a customer, a 
potential customer. This kind of analysis may also reveal reasons that lead to customers not setting up term deposits, which 
in some cases may be resolved by the Bank.

- #### How to classify the customers?
Before we train our classifiers, the data needs to be preprocessed. There are instances in a few categorical variables, where 
the value is unknown. Such instances are either removed or imputed with the highest occurring element relevant to a 
particular factor. After treating all such instances, behavior of each of the variables is analyzed and relationships between 
the variables is explored. 
For instance, if the duration of call made to a customer is long, then the customer is likely to set up a term deposit. In 
addition to this, if the outcome of the previous marketing campaign of a customer was a success, there are higher chances 
that the customer would convert this time as well i.e., set up a term deposit. Along with these, if the customer has a high 
consumer confidence index, the probability of conversion might be higher. All such relevant features are identified, and the 
customers are classified accordingly using suitable classification algorithms.



## Methods
- ### Exploratory Data Analysis
Exploratory Data Analysis (EDA) helps us in analyzing the data to visually summarize their characteristics. It helps us to see 
what the data can tell us beyond the formal modelling or statistical analysis task. Here, we have performed univariate and 
bivariate analysis to understand and interpret the different kinds of patterns that exist within the data. We have also 
inspected the data for any missing/null/unknown values during the EDA.

![image](https://github.com/pranavbansal04/Identifying-Potential-Customers/blob/main/plots/plot0.1.png?raw=true)
![image](https://github.com/pranavbansal04/Identifying-Potential-Customers/blob/main/plots/plot0.2.png?raw=true)