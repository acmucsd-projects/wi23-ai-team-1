import pandas as pd


def main():
    df = pd.read_csv("../input/train.csv")
    print(df)
    df = filterLength(df, 10)


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def filterLength(df, length: int):
    df["len"] = df["comment_text"].str.len()
    # print(df.loc[(df["len"] <= length) & (df["toxic"] == 0) & (
    #             df["severe_toxic"] == 0) & (df["obscene"] == 0) & (
    #                          df["threat"] == 0) & (df["insult"] == 0) & (
    #                          df["identity_hate"] == 0)])


    c = 0
    for i, r in df.iterrows():
        text = r["comment_text"]
        e = isEnglish(text)

        if not e:
            print(text)
            break
            c += 1
            if c % 1000 == 0:
                print(text)
    print(c)

    # df.DB_user.replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)
    df["comment_text"].replace({r'[^\x00-\x7F]+'}, regex=True, inplace=True)
    print(df.loc[df["id"] == "00025465d4725e87"])
    df.to_csv("../input/train_filtered.csv", index=False)

    return df.loc[df["len"] > length]


main()
