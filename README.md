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


**3. StandardScaler Transform**

3.1 We can apply the StandardScaler to the Sonar dataset directly to standardize the input variables.

3.2 We will use the default configuration and scale values to subtract the mean to center them on 0.0 and divide by the standard deviation to give the standard deviation of 1.0.


**4.Feature Selection method**

Remove less redundant features to overcome cluster overlapping.

**4.1 SelectKbest Method and using StratifiedKfold (n = 5)**

4.1.1 Select top 10 features and implemented on KNN and SVM model but it won't work out with good accuracy.

![SelectKBest graph](https://user-images.githubusercontent.com/60669591/122646946-715d0b00-d13f-11eb-88e3-9464514fa713.png)

**4.2 Pearson correlation Method using StratifiedKfold (n = 5)**

4.2.1 Select features having threshhold > 0.25 and implemented on KNN and SVM model but it won't work out with better accuracy.

![Correlation graph](https://user-images.githubusercontent.com/60669591/122647217-bb92bc00-d140-11eb-93eb-2f0227d04cc4.png)


**5. Use Xg-Boost model with StratifiedKfold (n = 5)**

5.1 Visualize features importances using Xg-Boost model.

![Xgboost graph](https://user-images.githubusercontent.com/60669591/122647473-0f51d500-d142-11eb-990e-46dc3882bff2.png)

![Xgboost graph1](https://user-images.githubusercontent.com/60669591/122647477-1547b600-d142-11eb-9636-93aea4296950.png)

5.2 After training the model on Xg-Boost. I got a classification report and it has far better accuracy as compared to K-NN and SVM model.

![XGBOOST for Binary Classification Model](https://user-images.githubusercontent.com/60669591/122647623-a28b0a80-d142-11eb-82fd-b3cc302fefc2.jpg)

**6. AtLast, Used Simple FCN model and Simple FCN model  with StratifiedKfold (n = 5)**

6.1 Training Results and Classification report of both models were mentioned below.

6.2 we can compare the results with theirs images.

![ANN2](https://user-images.githubusercontent.com/60669591/122647924-4d4ff880-d144-11eb-9017-1453dc55da8a.jpg)

![simple ANN k-fold](https://user-images.githubusercontent.com/60669591/122647730-470d4c80-d143-11eb-94a6-91691f5d1c44.jpg)

![simple ANN](https://user-images.githubusercontent.com/60669591/122647733-4aa0d380-d143-11eb-900c-372576d7aad8.jpg)

7. In github repository, there are 3 folders

7.1 CODE folder: Consist of train.py, eval.py, and CONFIG folder.

7.1.1 CONFIG folder: Consist of train_config.json and inference_config.json files. We have changed to change the directory path of training_data, testing_data, saving and loading model from a directory and, Checkpoint directory to run train.py and eval.py file.

7.2 MODEL folder: Consist of saved_model and Checkpoint directory.

7.3 DATA folder: Consist of training_csv and test_csv file.




 
