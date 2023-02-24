import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


def graphBasedOnNumWords(df: pd.DataFrame) -> None:
    """
    Graphs the number of words in each comment.
    :param df: The Pandas DataFrame.
    :return: None.
    """
    df["num_words"] = df["comment_text"].apply(lambda x: len(x.split()))
    df["num_words"].hist(bins=30, range=(0, 300),
                         weights=np.ones(len(df)) / len(df))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.title("Number of Words in Each Comment")
    plt.xlabel("Number of Words")
    plt.ylabel("Number of Comments")
    plt.show()
    plt.close()
    print(df["num_words"].describe())


def getFrequencyOfWords(df: pd.DataFrame) -> None:
    """
    Gets the frequency of words in all comment.
    :param df: The Pandas DataFrame.
    :return: None.
    """
    print(pd.Series(" ".join(df["comment_text"]).split()).value_counts()[:20])


def getFrequencyOfToxicWords(df: pd.DataFrame) -> None:
    """
    Gets the frequency of words in all comment.
    :param df: The Pandas DataFrame.
    :return: None.
    """

    toxicValues = ["toxic", "severe_toxic", "obscene", "threat", "insult",
                   "identity_hate"]
    filteredDf = df.loc[df[toxicValues].sum(axis=1) > 0]

    print(pd.Series(" ".join(filteredDf["comment_text"]).split())
          .value_counts()[:20])
