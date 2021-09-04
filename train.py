
import pandas as pd
import numpy as np
import logging


from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train, compute_model_metrics, inference, save_to_file, load_file

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

def go():
    df = pd.read_csv("data/census_cleaned.csv")

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    logging.info(f'Training sample size: {len(df_train)}')
    logging.info(f'Testing sample size: {len(df_test)}')

    X_train, y_train, encoder, lb = process_data(df_train, CAT_FEATURES, 'salary')
    save_to_file(encoder, f'model/encoder.pkl')
    save_to_file(lb, f'model/lb.pkl')

    X_test, y_test, _, _ = process_data(
        df_test, CAT_FEATURES, 'salary', False, encoder, lb)

    lr = train(X_train, y_train)
    save_to_file(lr, f'model/lr.pkl')

    y_pred = inference(lr, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
    logging.info('Metrics on Testing data:')
    logging.info(f'Precision: {precision}, Recall: {recall}, fbeta: {fbeta}')


    for feature in CAT_FEATURES:

        data = pd.DataFrame({feature: df_test[feature],
                            'salary': y_test,
                            'pred': y_pred})
        with open(f'model/slice_output.txt', 'a') as f:
            f.write(f'----------- {feature} ----------\n')

            for val in data[feature].unique():
                tmp_df = data[data[feature] == val]
                precision, recall, fbeta = compute_model_metrics(
                    tmp_df['salary'], tmp_df['pred'])
                f.write(f'{val} - Precision: {str(round(precision, 2))}, Recall: {str(round(recall,2))}, fbeta: {str(round(fbeta, 2))}.\n')

if __name__ == '__main__':
    go()