import streamlit as st
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import StratifiedShuffleSplit


WILL_RUN = True
TOXIC_CATEGORIES = ["toxic"]

data = pd.read_csv(
    "/kaggle/input/toxic-message-classifier-dataset/train_cleaned.csv")


def createEncoder():
    NUM_ROWS = 150000
    BATCH_SIZE = 64

    MAX_TOKENS = 5000
    MAX_LENGTH = 200
    encoder = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS, output_sequence_length=MAX_LENGTH)
    encoder.adapt(data.head(NUM_ROWS)["comment_text"].tolist())

#     vocab = np.array(encoder.get_vocabulary())
#     vocab[:20]

    return encoder


def createModel(encoder):
    # Sets random seed so results are identical every time
    SEED = 1
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=256,
            mask_zero=True
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(128, 5, padding="valid",
                               activation="relu", strides=1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(TOXIC_CATEGORIES), activation="sigmoid")
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=["accuracy"])

    return model


def getTrainingData():
    # Classification for all types of toxicity
    multiDf = data.head(NUM_ROWS)[["comment_text"] + TOXIC_CATEGORIES]

    x = multiDf["comment_text"]
    y = multiDf[TOXIC_CATEGORIES]

    splitter = StratifiedShuffleSplit(random_state=1, test_size=0.2)

    for train, test in splitter.split(x, y[TOXIC_CATEGORIES[0]]):
        training_data = x.iloc[train]
        training_target = y.iloc[train]
        validation_data = x.iloc[test]
        validation_target = y.iloc[test]

    return training_data, training_target, validation_data, validation_target


def trainModel(model, training_data, training_target, validation_data, validation_target):
    # Early Stopping
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=2)

    history = model.fit(training_data, training_target, epochs=10, validation_data=(
        validation_data, validation_target), callbacks=[callback], batch_size=32)


if WILL_RUN:
    encoder = createEncoder()
    model = createModel(encoder)
    td, tt, vd, vt = getTrainingData()
    trainModel(model, td, tt, vd, vt)


def getTestData():
    test_data = pd.read_csv(
        "/kaggle/input/toxic-message-classifier-dataset/test.csv")
    test_labels = pd.read_csv(
        "/kaggle/input/toxic-message-classifier-dataset/test_labels.csv")

    merged_df = test_labels.merge(test_data, left_on="id", right_on="id")
    merged_df = merged_df.loc[(merged_df["toxic"] != -1) & (merged_df["severe_toxic"] != -1) & (merged_df["obscene"] != -1)
                              & (merged_df["threat"] != -1) & (merged_df["insult"] != -1) & (merged_df["identity_hate"] != -1)]

    return merged_df


def testZeroValues(merged_df):
    test_df = merged_df["comment_text"]
    query = " & ".join([f"({label} == 0)" for label in TOXIC_CATEGORIES])
    filtered_df = merged_df.query(query)

    filtered_test_dataset = filtered_df["comment_text"]
    filtered_df_target = filtered_df[TOXIC_CATEGORIES]
    model.evaluate(filtered_test_dataset, filtered_df_target)


def testOneValues(merged_df):
    test_df = merged_df["comment_text"]
    query = " | ".join([f"({label} == 1)" for label in TOXIC_CATEGORIES])
    filtered_df = merged_df.query(query)

    filtered_test_dataset = filtered_df["comment_text"]
    filtered_df_target = filtered_df[TOXIC_CATEGORIES]
    model.evaluate(filtered_test_dataset, filtered_df_target)


def testAllValues(merged_df):
    test_df = merged_df["comment_text"]
    test_target = merged_df[TOXIC_CATEGORIES]
    model.evaluate(test_df, test_target)


if WILL_RUN:
    merged_df = getTestData()
    testAllValues(merged_df)
    testZeroValues(merged_df)
    testOneValues(merged_df)


def getSubmissionFile():
    submission_set = pd.read_csv(
        '/kaggle/input/toxic-message-classifier-dataset/test.csv')
    submission_set.head()

    x_test = submission_set['comment_text'].values
    y_testing = model.predict(x_test, verbose=1, batch_size=BATCH_SIZE)

    submission_df = pd.DataFrame(columns=[
                                 'id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
    submission_df['id'] = submission_set['id']
    submission_df['toxic'] = [0 if x[0] < 0.5 else 1 for x in y_testing]
    submission_df['severe_toxic'] = [0 if x[1] < 0.5 else 1 for x in y_testing]
    submission_df['obscene'] = [0 if x[2] < 0.5 else 1 for x in y_testing]
    submission_df['threat'] = [0 if x[3] < 0.5 else 1 for x in y_testing]
    submission_df['insult'] = [0 if x[4] < 0.5 else 1 for x in y_testing]
    submission_df['identity_hate'] = [
        0 if x[5] < 0.5 else 1 for x in y_testing]

    submission_df.head()

    submission_df.to_csv('/kaggle/working/submission.csv', index=False)


df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})
