import logging
import os

import pandas as pd
from catboost import CatBoostClassifier


# Настройка логгера
logger = logging.getLogger(__name__)

logger.info('Importing pretrained model...')

# Import model
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model = CatBoostClassifier()
model.load_model('./models/my_catboost.cbm')

# Define optimal threshold
THRESHOLD = 0.5
logger.info('Pretrained model imported successfully...')


def top_features():
    feature_importance = model.get_feature_importance()
    feature_names = model.feature_names_

    importances_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance,
    })

    top_5_features = importances_df.sort_values(by='importance', ascending=False).head(5)

    return dict(zip(top_5_features['feature'], top_5_features['importance'], strict=False))


# Make prediction
def make_pred(dt, path_to_file):
    y_proba = model.predict_proba(dt)[:, 1]

    # Make submission dataframe
    submission = pd.DataFrame({
        'index': pd.read_csv(path_to_file).index,
        'prediction': (y_proba > THRESHOLD) * 1,
    })
    logger.info('Prediction complete for file: %s', path_to_file)

    top_5_features = top_features()
    logger.info('Got top 5 features')

    # Return proba for positive class
    return submission, top_5_features, y_proba
