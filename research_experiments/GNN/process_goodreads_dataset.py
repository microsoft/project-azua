import pandas as pd
import os
import numpy as np
import re

fns = os.listdir("./")
book_files = [v for v in fns if "book" in v]
user_files = [v for v in fns if "user" in v]

test_ratio = 0.1
val_ratio = 0.1

data_dir = "/PATH/TO/PROJECT/data/goodreads/"
metadata_dir = os.path.join(data_dir, "metadata")


def remove_tags(string):
    result = re.sub("<.*?>", "", string)
    return result

booknames_from_book = np.array([])
for i, book_file in enumerate(book_files):
    df = pd.read_csv(book_file)
    if "Description" not in df:
        continue
    df = df[~df.Description.isna()]
    if "Language" not in df:
    df = df[df.Language.isin(["eng", "en-US", "en-GB", np.nan])]
    if "Name" in df:
        assert "name" not in df
        booknames_from_book = np.union1d(booknames_from_book, df.Name.unique())
    else:
        assert "name" in df
        booknames_from_book = np.union1d(booknames_from_book, df.name.unique())


booknames_from_user = np.array([])
for i, user_file in enumerate(user_files):
    df = pd.read_csv(user_file)
    if "Name" in df:
        assert "name" not in df
        booknames_from_user = np.union1d(booknames_from_user, df.Name.unique())
    else:
        assert "name" in df
        booknames_from_user = np.union1d(booknames_from_user, df.name.unique())

booknames = set(booknames_from_book) & set(booknames_from_user)


booknames_from_book = np.array([])
cnt = 0
for i, book_file in enumerate(book_files):
    df = pd.read_csv(book_file)
    if "Description" not in df:
        continue
    if "name" in df:
        df = df.rename(columns={"name": "Name"})

    df = df[["Name", "Description"]]
    df = df[df["Name"].isin(booknames)]
    if cnt == 0:
        df_books = df
    else:
        df_books = pd.concat((df_books, df), ignore_index=True)
    cnt += 1

df_books = df_books[~df_books.Description.isna()]
df_books = df_books.loc[df_books.Name.drop_duplicates().index]
df_books = df_books.reset_index(drop=True)

assert len(df_books) == len(booknames)

ratings_map = {
    "did not like it": 0,
    "it was ok": 1,
    "liked it": 2,
    "really liked it": 3,
    "it was amazing": 4,
    "This user doesn't have any rating": -1,
}

cnt = 0
for i, user_file in enumerate(user_files):
    df = pd.read_csv(user_file)
    if "name" in df:
        df = df.rename(columns={"name": "Name"})

    if "ID" in df:
        df = df.rename(columns={"ID": "Id"})
    df.Rating = df.Rating.map(ratings_map)
    assert len(df[df.Rating.isna()]) == 0
    df.Rating = df.Rating / df.Rating.max()

    df = df[df["Name"].isin(booknames)]

    if cnt == 0:
        df_all = df
    else:
        df_all = pd.concat((df_all, df), ignore_index=True)
    cnt += 1

assert len(df_all.Name.unique()) == len(booknames)

df_all = df_all.groupby("Id").filter(lambda x: len(x) > 10)
df_all = df_all.groupby("Name").filter(lambda x: len(x) > 10)

bookname_id_map = {v: i for i, v in enumerate(sorted(df_all.Name.unique()))}
username_id_map = {v: i for i, v in enumerate(sorted(df_all.Id.unique()))}

df_all.Name = df_all.Name.map(bookname_id_map)
df_all.Id = df_all.Id.map(username_id_map)

df_all = df_all.rename(columns={"Id": "row", "Name": "col", "Rating": "val"})

df_all = df_all.drop_duplicates()
df_all = df_all.reset_index(drop=True)

df_books.Name = df_books.Name.map(bookname_id_map)
df_books = df_books[~df_books.Name.isna()]
df_books.Name = df_books.Name.astype(np.int)

df_books = df_books.sort_values(by=["Name"])
df_books = df_books.reset_index(drop=True)

df_books.Description = df_books.Description.apply(lambda x: remove_tags(x))

if not os.path.exists(data_dir):
    os.mkdir(data_dir)
    os.mkdir(metadata_dir)
df_all.to_csv(os.path.join(data_dir, "all.csv"), index=None, header=None)

# pd.read_csv(os.path.join(data_dir, 'all.csv'), header=None, names=["row", "col", "val"])
df_books.to_csv(os.path.join(metadata_dir, "sentences.csv"), index=None)

while True:
    train_idxs = np.random.choice(df_all.index, size=int(len(df_all) * (1 - test_ratio - val_ratio)), replace=False)
    df_train = df_all.loc[train_idxs]

    if len(df_train.row.unique()) == len(df_all.row.unique()) and len(df_train.col.unique()) == len(
        df_all.col.unique()
    ):
        break

df_train.to_csv(os.path.join(data_dir, "train.csv"), index=None, header=None)

rest_idxs = np.setdiff1d(df_all.index, train_idxs)

test_idxs = np.random.choice(rest_idxs, size=int(len(rest_idxs) * test_ratio / (test_ratio + val_ratio)), replace=False)
val_idxs = np.setdiff1d(rest_idxs, test_idxs)

df_all.loc[test_idxs].to_csv(os.path.join(data_dir, "test.csv"), index=None, header=None)
df_all.loc[val_idxs].to_csv(os.path.join(data_dir, "val.csv"), index=None, header=None)
