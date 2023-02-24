import pandas as pd

from cleaning import filterNonEnglishChars, filterByNumWords, splitIntoWords, \
    toLowerCase, removeStopWords, trimWhitespace
from src.exploring import graphBasedOnNumWords, getFrequencyOfWords, \
    getFrequencyOfToxicWords


def cleanData():
    df = pd.read_csv("input/train.csv")
    df = toLowerCase(df)
    df = filterNonEnglishChars(df)
    df = removeStopWords(df)
    df = trimWhitespace(df)
    df = splitIntoWords(df)
    df = filterByNumWords(df, 3)

    df.to_csv("input/train_cleaned.csv", index=False)


def main():
    # cleanData()

    df = pd.read_csv("input/train_cleaned.csv")
    # graphBasedOnNumWords(df)
    # getFrequencyOfWords(df)
    getFrequencyOfToxicWords(df)


main()
