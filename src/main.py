import pandas as pd

from cleaning import filterNonEnglishChars, filterByNumWords, splitIntoWords, \
    toLowerCase, removeStopWords, trimWhitespace
from exploring import graphBasedOnNumWords, getFrequencyOfWords, \
    getFrequencyOfToxicWords
from modeling import build_dummy_model


def cleanData():
    # df = pd.read_csv("input/train.csv")
    df = toLowerCase(df)
    df = filterNonEnglishChars(df)
    df = removeStopWords(df)
    df = trimWhitespace(df)
    df = splitIntoWords(df)
    df = filterByNumWords(df, 3)

    # df.to_csv("input/train_cleaned.csv", index=False)


def main():
    # cleanData()

    # df = pd.read_csv("input/train_cleaned.csv")
    # graphBasedOnNumWords(df)
    # getFrequencyOfWords(df)
    # getFrequencyOfToxicWords(df)
    # cleaned_df = pd.read_csv("input/train_cleaned.csv")
    # print(cleaned_df.shape)
    # print(cleaned_df.head())
    build_dummy_model()


main()
