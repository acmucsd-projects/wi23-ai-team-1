import pandas as pd

from cleaning import filterNonEnglishChars, filterByNumWords, splitIntoWords, \
    toLowerCase, removeStopWords, trimWhitespace


def main():
    df = pd.read_csv("input/train.csv")
    print(df.head(5)["comment_text"])
    df = toLowerCase(df)
    df = filterNonEnglishChars(df)
    df = removeStopWords(df)
    df = trimWhitespace(df)
    df = splitIntoWords(df)
    df = filterByNumWords(df, 3)

    print(df.head(5))


main()
