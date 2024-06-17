Heart Disease Prediction Project
This project focuses on predicting heart disease using various machine learning models. It includes data preprocessing, visualization, feature engineering, model training, hyperparameter tuning, and evaluation.

Table of Contents
Introduction
Setup
Data Preprocessing
Visualization
Feature Engineering
Model Training and Evaluation
Results
Usage
Conclusion

Introduction
The goal of this project is to predict the presence of heart disease using a dataset that includes various medical attributes. The project utilizes different machine learning algorithms, evaluates their performance, and selects the best model for prediction.

Setup
To run this project, you need Python installed along with the following packages:

numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
Install the necessary packages using:
pip install numpy pandas matplotlib seaborn scikit-learn scipy

Data Preprocessing
The dataset is loaded and processed to handle missing values, convert categorical features, and normalize continuous features. The code:

Converts specific features to the appropriate data types.
Checks for and handles missing values.
Applies transformations to continuous features.
Visualization
The code generates visualizations to understand the distribution of continuous and categorical features:

Histograms and KDE plots for continuous features.
Bar plots for categorical features.
Feature Engineering
One-hot encoding is applied to categorical features, and continuous features are transformed using the Box-Cox method to stabilize variance and make the data more Gaussian-like.

Model Training and Evaluation
The project uses the following machine learning models:

Decision Tree
Random Forest
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Logistic Regression
Hyperparameter tuning is performed using GridSearchCV for optimal model performance. The models are evaluated based on recall, precision, F1-score, and accuracy.

Results
The models' performances are compared, and the results are visualized in a horizontal bar chart. The best-performing model is used for final predictions.

Usage
To use the model for predictions:

Train the model using the provided code.
Input new data in the specified format.
Run the prediction function to get results.
Example:
Input_data = (63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0)
prediction = pipeline.predict(np.asarray(input_data).reshape(1, -1))

Conclusion
This project demonstrates a comprehensive approach to building a machine learning model for heart disease prediction, covering data preprocessing, visualization, feature engineering, model training, and evaluation. The best model can be used for future predictions, providing valuable insights into heart disease diagnosis.
