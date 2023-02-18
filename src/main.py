import pandas as pd


def main():
    df = pd.read_csv("../input/train.csv")
    print(df)
    df = filterNonEnglishChars(df)
    df = filterLength(df, 10)
    print(df)


def isEnglish(s: str) -> bool:
    """
    Check if a string contains all english characters

    :param s: string to check
    :return: whether string contains only english characters
    """
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def filterLength(df: pd.DataFrame, length: int) -> pd.DataFrame:
    """
    Filter out comments that are less than a certain length

    :param df: dataframe
    :param length: int
    :return: dataframe with comments that are greater than the length
    """

    df["len"] = df["comment_text"].str.len()
    # Could also check if all toxic columns are 0
    # print(df.loc[(df["len"] <= length) & (df["toxic"] == 0) & (
    #         df["severe_toxic"] == 0) & (df["obscene"] == 0) & (
    #                      df["threat"] == 0) & (df["insult"] == 0) & (
    #                      df["identity_hate"] == 0)])

    return df.loc[df["len"] > length]


def filterNonEnglishChars(df: pd.DataFrame) -> pd.DataFrame:
    """
        Filter out non-english characters

        :param df: dataframe
        :return: dataframe with non-english characters filtered out
    """

    df["comment_text"] = df['comment_text'].str \
        .encode('ascii', 'ignore').str.decode('ascii')

    return df


main()
