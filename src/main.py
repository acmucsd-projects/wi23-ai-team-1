import pandas as pd

from cleaning import filterNonEnglishChars, filterByNumWords, splitIntoWords, \
    toLowerCase


def main():
    df = pd.read_csv("../input/train.csv")
    print(df)
    df = toLowerCase(df)
    df = filterNonEnglishChars(df)
    df = splitIntoWords(df)
    df = filterByNumWords(df, 3)
    print(df)


main()
