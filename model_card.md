# Model Card

## Model Details
Gejun Zhu created this logistic regression model with default hyperparameters in scikit-learn 0.24.2.

## Intended Use
This model predicts whether the salary of a given person will be over 50k or not. 

## Training Data
The data was obtained from UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income).

## Evaluation Data
The original was splitted into 80/20. 80% was used for training, and the rest 20% was used for evaluation. 


## Metrics
- Precision: 0.7285
- Recall: 0.2699
- fbeta: 0.3939

## Ethical Considerations
The data was obtained from public census in 1994. There is no any decision making based on this prediction. Hence, there is not ethical concerns. 

## Caveats and Recommendations
- Would boost model performance when using more complex model
- Feature engineering such as creating new features