import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def text_vectorization(data):
    maxlen = 0
    longest_comment = ""
    for comment in data['comment_text']:
        length = len(comment)
        if (length > maxlen):
            longest_comment = comment
    maxlen = max(maxlen, length)

    vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(

        max_tokens=None,

        # this is greater than the max words any comment has (774)
        # the remaning spots in the output would be padded by 0s
        output_sequence_length=800,

        # converets to lowercase and skips all the punctuation
        standardize="lower_and_strip_punctuation",

        # the tokens will be split at whitespaces
        split="whitespace",

        # each of the tokens is represented as an integer
        output_mode="int",
    )

    numpyArray = data[data.columns[1]].to_numpy()
    vectorize_layer.adapt(numpyArray)

    return vectorize_layer


def build_poop_model():
    df = pd.read_csv("input/train_cleaned.csv")

    X = df["comment_text"]
    y = df["toxic"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print(X_train.head())
    print(y_train.head())

    vectorized_layer = text_vectorization(X_train)

    model = tf.keras.Sequential()
    model.add(vectorized_layer)
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    y_hat = model.predict(X_test)
    y_hat = [1 if y >= 0.5 else 0 for y in y_hat]

    print(accuracy_score(y_test, y_hat))
