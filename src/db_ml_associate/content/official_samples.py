"""Stems of the five official sample questions in the exam guide (2025-03-01).

Used only to detect copies: book questions must be original (FR-019), so ``book check`` flags any
question whose stem is too similar to one of these.
"""

OFFICIAL_SAMPLE_STEMS: tuple[str, ...] = (
    "A data scientist wants to create a feature table to use in their models. They are working in a "
    "workspace with Unity Catalog enabled and want this feature table to be stored and governed by it. "
    "What is the correct way of creating this feature table?",
    "A data scientist needs to impute the missing values in a continuous feature. They want to do this "
    "with the least amount of effort but with correct results. Which strategy will do this?",
    "A data scientist is working on a machine learning project to develop a model that predicts whether "
    "a customer will churn from a subscription service. The dataset is highly imbalanced, with only 10% "
    "of the instances representing customers who churn. They want to ensure that your model effectively "
    "identifies the minority class without being biased towards the majority class. Which strategy "
    "directly mitigates the model's bias towards the non-churn customers due to class imbalance?",
    "A data scientist is tuning a Support Vector Machine (SVM) model using 5 fold cross-validation and "
    "GridSearchCV in scikit-learn. The parameter grid includes three hyperparameters to optimize: C with "
    "values [0.1, 1, 10], kernel with choices ['linear', 'rbf'], and gamma with values [0.01, 0.1, 1]. "
    "How many different models will be trained in total?",
    "A company has a podcast platform that has thousands of users. The company has implemented an anomaly "
    "detection algorithm to detect low podcast engagement based on a 10-minute running window of user "
    "events such as listening, pausing, and exiting the podcast. A machine learning engineer wants to "
    "deploy this model into a production data pipeline that needs to handle up to tens of thousands of "
    "events per second. As the volume of events fluctuates throughout the day, the engineer needs the "
    "pipeline compute to be resized dynamically. Which pipeline design approach meets these requirements?",
)

SIMILARITY_THRESHOLD = 0.85
