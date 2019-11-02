Boston-House-Price-dataset

The Boston Housing Dataset consists of price of houses in various places in Boston. Alongside with price, the dataset also provide information such as Crime (CRIM), areas of non-retail business in the town (INDUS), the age of people who own the house (AGE), and there are many other attributes that available here.

The dataset is available at kaggle https://www.kaggle.com/c/boston-housing

In this post, we are going to learn how to implement linear regression on Boston Housing dataset using scikit-learn.
Boston Housing Dataset. When our output is real or continuous value the problem is considered as regression problem. Example predicting ‘price’, ‘weight’, 'projected_units_of_sale' etc.

We can apply various regression models to solve this, like SVM, Linear Regression etc. 

We see how linear Regression works.
	Linear Regression to find best hyper-plane which goes through the points. We need to optimise the LR model, when many points lie much above or below the hyperplane. This is called optimisation . We are going to manually implement and optimize SGD for Linear Regression

1) Manual implementation of SGD for linear regression 

Our aim is to find the best Weight vector W that contains--> {W1,W2..Wk} correspending to 'k' features of our dataset, and constant b. 

Optimization equation of LR is:- 
best (W , b) =1/n(argmin Σ (y(actual)-y(pred))^2)

The Equation for linear regression :-

L(W,B)= 1/n Σ (y-w.x-b)^2 where y(pred) = w.x and b is a constant and X-> our input dataset

We take partial derivative for both W and B

dL/dW = 1/n Σ (-2x)(y-w.x-b)
dL/dB= 1/n Σ (-2)(y-w.x-b)
W(j+1)=W(j) – alpha *(dL/dW)
B(j+1)=B(j) – alpha *(dL/dB)

alpha-> Hyperparameter

After finding the best B and W. Substite it in the equation L(W,B)= 1/n Σ (y-w.x-b)^2


2) comparing the results of SGD regressor of sklearn and our manual implementation on Boston home price dataset

