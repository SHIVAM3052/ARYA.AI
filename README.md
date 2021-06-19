**# ARYA.AI**
**Binary Classification**

**1. Dataset contain 2 CSV file.**

1.1 Training CSV file : Consist of 57 features and Y as label in (0/1)

1.2 Test CSV file : Consist of 57 features only 

I had used four models for binary classifications 

a. K-NN

b. SVM

c. Xg-Boost

d. FCN

**2. Data visualisation using t-sne**

![tsne_graph](https://user-images.githubusercontent.com/60669591/122646518-2c37d980-d13d-11eb-91f9-bfb8ba22a916.png)

Non-linear technique for dimensionality reduction that is particularly well suited for the visualization of high-dimensional datasets. In the above t-sne graph, we can visualize feature distributions of all data points. We have find patterns in the data by identifying observed clusters based on similarity of data points with multiple features.

2.1 Use feature engineering method to remove overlap features but it didn't work out.

2.2 K-NN : It can not handle overlap features and outliers. It is not suitable to identify clusters with non-convex shapes as shown in t-sne Graph

2.3 SVM :  SVM does not perform very well in our data set has more noise i.e. target classes are overlapping. Our model able to give less accuracy due to clusters overlapping.

2.4 Xg-Boost : When there is a larger number of training samples. Ideally, greater than 1000 training samples and less 100 features or we can say when the number of features <       number of training samples. When there is a mixture of categorical and numeric features or just numeric features. Model yields better result on our dataset.

2.5 FCN : Implementin simple FCN model with single Dense layer and using activation 'relu'. At last layer we use sigmoid function for binary classification. Model yields better     result on our dataset.

**FROM HERE STARTED MY BINARY CLASSIFICATION JOURNEY ON KNN & SVM MODEL WITH FEATURE ENGINEERING TO PREDICT BETTER RESULTS**

**3. StandardScaler Transform**

3.1 I had apply the StandardScaler to the training dataset directly to standardize the all input features.

3.2 Use SKLEARN preprocessing StandardScaler as default configuration and scale values to subtract the mean to center them on 0.0 and divide by the standard deviation to give       the standard deviation of 1.0. It is useful for data which has negative values. It also arranges the data in a standard normal distribution.


**4.Feature Selection method**

It is used for feature selection/dimensionality reduction on sample sets, either to improve estimators’ accuracy scores or to boost their performance on very high-dimensional datasets.

a. Reduces Overfitting: Less redundant data means less opportunity to make decisions based on noise.

b. Improves Accuracy: Less misleading data means modeling accuracy improves.

c. Reduces Training Time: Less data means that algorithms train faster.

**4.1 SelectKbest Method and using StratifiedKfold (n = 5)**

4.1.1 f_classif: I have used only for categorical targets and based on the Analysis of Variance (ANOVA) statistical test.

4.1.2 Select top 10 features.  

4.1.3 Apply StratifiedKfold (n = 5) shuffles your data, after that splits the data into n_splits parts and Done. Now, it will use each part as a test set.I had implemented it on       KNN and SVM and trained our model but it won't work out with good accuracy.

![SelectKBest graph1](https://user-images.githubusercontent.com/60669591/122656566-498a9900-d179-11eb-962d-34fda3588cd7.png)

**4.2 Pearson correlation Method using StratifiedKfold (n = 5)**

4.2.1 I thought to check correlation between features and used Pearson correlation Method using StratifiedKfold (n = 5)

4.2.1 Select features having threshhold > 0.25 and implemented on KNN and SVM model. It yields better result as compared to SelectKbest method but it won't work out with         better accuracy.

4.2.3 In below chart, diagonals are all 1/dark green because those squares are correlating each variable to itself (so it's a perfect correlation). For the rest the larger the         number and darker the color the higher the correlation between the two variables.

![Correlation graph](https://user-images.githubusercontent.com/60669591/122647217-bb92bc00-d140-11eb-93eb-2f0227d04cc4.png)


**AFTER TRYING ABOVE METHOD ON KNN AND SVM NOT YIELD GOOD RESULT. SO DECIDED TO SHIFT ON XG-BOOST AND FCN MODEL**

**5. Use Xg-Boost model with StratifiedKfold (n = 5)**

5.1 Visualize features importances using Xg-Boost model. Feature importance provides a score that indicates how useful or valuable each feature was in the construction of the       boosted decision trees within the model.

![Xgboost graph2](https://user-images.githubusercontent.com/60669591/122656541-23fd8f80-d179-11eb-8157-8d1026d19f70.png)

![Xgboost graph1](https://user-images.githubusercontent.com/60669591/122647477-1547b600-d142-11eb-9636-93aea4296950.png)

5.2 After training the model on Xg-Boost. I got a classification report and it has far better accuracy as compared to K-NN and SVM model. Calculate classification report mentioned in below screen shot. 

5.3 As you see in confusion matrix false Negative > false positive i.e Misclassification is not balanced and bias toward particular class

5.4 Misclassification Rate : (F.P + F.N) / (T.P + F.P + T.N + F.N) = (13 + 31) / ( 585 + 13 + 349 + 31 ) = 0.044

![XGBOOST for Binary Classification Model](https://user-images.githubusercontent.com/60669591/122647623-a28b0a80-d142-11eb-82fd-b3cc302fefc2.jpg)


**6. AtLast, Used Simple FCN model and Simple FCN model  with StratifiedKfold (n = 5)**

6.1 Training Results and Classification report of both models were mentioned below.

6.2 we can compare the results with theirs images.

![ANN2](https://user-images.githubusercontent.com/60669591/122647924-4d4ff880-d144-11eb-9017-1453dc55da8a.jpg)

![simple ANN k-fold](https://user-images.githubusercontent.com/60669591/122647730-470d4c80-d143-11eb-94a6-91691f5d1c44.jpg)

![simple ANN](https://user-images.githubusercontent.com/60669591/122647733-4aa0d380-d143-11eb-900c-372576d7aad8.jpg)

6.3 Misclassification Rate of FCN with StratifiedKfold : (F.P + F.N) / (T.P + F.P + T.N + F.N) = (29 + 25) / ( 573 + 29 + 351 + 25 ) = 0.055

6.4 Misclassification Rate of FCN : (F.P + F.N) / (T.P + F.P + T.N + F.N) = (26 + 25) / ( 576 + 26 + 351 + 25 ) = 0.054

**7. In github repository, there are 3 folders**

7.1 CODE folder: Consist of train.py, eval.py, and CONFIG folder.

7.1.1 CONFIG folder: Consist of train_config.json and inference_config.json files. We have changed to change the directory path of training_data, testing_data, saving and loading model from a directory and, Checkpoint directory to run train.py and eval.py file.

7.2 MODEL folder: Consist of saved_model and Checkpoint directory.

7.3 DATA folder: Consist of training_csv and test_csv file.


**8. lIBRARY USED**

a. PANDAS

b. NUMPY

c. sklearn.preprocessing.StandardScaler

d. sklearn.model_selection.StratifiedKFold

e. sklearn.metrics.classification_report

f. sklearn.metrics

g. TENSORFLOW

h. KERAS

i. SEABORN

j. MATPLOTLIB


 
