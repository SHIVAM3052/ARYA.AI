**# ARYA.AI**
**Binary Classification**

**1. Dataset contain 2 CSV file.**

1.1 Training CSV file : Consist of 57 features and Y as label in (0/1)

1.2 Test CSV file : Consist of 57 features only 



**2. Data visualisation using t-sne**

![tsne_graph](https://user-images.githubusercontent.com/60669591/122646518-2c37d980-d13d-11eb-91f9-bfb8ba22a916.png)

In the above t-sne graph, we can visualize feature distributions of all data points.

2.1 Use feature engineering method to remove overlap features but it didn't work out.

2.2 K-NN and SVM model able to give less accuracy due to clusters overlapping.


**4.Feature Selection method**

Remove less redundant features to overcome cluster overlapping.

**4.1 SelectKbest Method and using StratifiedKfold (n = 5)**

4.1.1 Select top 10 features and implemented on KNN and SVM model but it won't work out with good accuracy.

![SelectKBest graph](https://user-images.githubusercontent.com/60669591/122646946-715d0b00-d13f-11eb-88e3-9464514fa713.png)

**4.2 Pearson correlation Method using StratifiedKfold (n = 5)**

4.2.1 Select features having threshhold > 0.25 and implemented on KNN and SVM model but it won't work out with better accuracy.

![Correlation graph](https://user-images.githubusercontent.com/60669591/122647217-bb92bc00-d140-11eb-93eb-2f0227d04cc4.png)


 
