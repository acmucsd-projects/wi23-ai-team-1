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
