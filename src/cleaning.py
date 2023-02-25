import pandas as pd
from nltk.corpus import stopwords


def filterByNumWords(df: pd.DataFrame, numWords: int) -> pd.DataFrame:
    """
    Filter out comments that have fewer words than numWords

    :param df: dataframe
    :param numWords: int
    :return: dataframe with comments with fewer words than numWords filtered out
    """

    if "comment_text_words" not in df.columns:
        df = splitIntoWords(df)

    return df.loc[df["comment_text_words"].str.len() > numWords]


def filterNonEnglishChars(df: pd.DataFrame) -> pd.DataFrame:
    """
        Filter out non-english characters

        :param df: dataframe
        :return: dataframe with non-english characters filtered out
    """

    # df["comment_text"] = df['comment_text'].str \
    #     .encode('ascii', 'ignore').str.decode('ascii')
    df["comment_text"].replace(r"[^A-Za-z\s]+", "", regex=True,
                               inplace=True)

    return df


def removeStopWords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove stop words

    :param df: dataframe
    :return: dataframe with stop words removed
    """

    wordsToRemove = stopwords.words('english')
    pattern = r"\b({})\b".format('|'.join(wordsToRemove))
    df["comment_text"] = df["comment_text"].str.replace(
        pattern, "", regex=True)

    return df


def toLowerCase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forces all characters to be lowercase

    :param df: dataframe
    :return: dataframe with all characters forced to be lowercase
    """

    df["comment_text"] = df["comment_text"].str.lower()

    return df


def splitIntoWords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split comments into words and forces them to be lowercase

    :param df: dataframe
    :return: dataframe with comments split into words
    """

    df["comment_text_words"] = df["comment_text"].str.split("\\s+")

    return df


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


def trimWhitespace(df: pd.DataFrame) -> pd.DataFrame:
    df["comment_text"] = df["comment_text"].str.strip()

    return df

customStopWords = stopwords.words("english")

def additionalStopWords():
    customStopWords.append('wikipedia')
    customStopWords.append('article')
    customStopWords.append('page')
