import os

import numpy as np
import pandas as pd

SUBMISSIONS_FOLDER = 'submissions'


def create_submission(predictions):
    index = 1
    image_ids = []
    labels = []
    for prediction in predictions:
        image_ids.append(index)
        labels.append(np.argmax(prediction))
        index += 1

    df_predictions = pd.DataFrame({'ImageId': image_ids, 'Label': labels})

    submissions = os.listdir(SUBMISSIONS_FOLDER)

    submissions_count = len(submissions)

    df_predictions.to_csv(
        f'{SUBMISSIONS_FOLDER}/submission_{submissions_count}.csv',
        index=False)
