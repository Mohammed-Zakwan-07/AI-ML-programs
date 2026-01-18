import pandas as pd
df = pd.read_csv("file.csv")
df = pd.read_excel("file.xlsx")
df = pd.read_json("data.json")
df.head()        # first 5 rows
df.tail()        # last 5 rows
df.info()        # column info & data types
df.describe()    # summary statistics
df.shape         # (rows, columns)
df.columns       # list of column names
df["Age"]                  # single column
df[["Name", "Age"]]        # multiple columns
df.iloc[0]                 # first row (index location)
df.iloc[0:5]               # first five rows
df.loc[df["Age"] > 30]     # filter rows
df.isnull().sum()               # check missing values
df.dropna()                     # drop rows with missing
df.fillna(0)                    # fill missing with 0
df["Age"] = df["Age"].fillna(df["Age"].mean())  # fill with mean
df["FullName"] = df["First"] + " " + df["Last"]
df["Age"] = df["Age"].astype(int)          # convert type
df.rename(columns={"old": "new"}, inplace=True)
df.sort_values("Age")
df.sort_values(["Age", "Salary"], ascending=False)
df.groupby("Gender")["Salary"].mean()
df.groupby("Department").sum()
df.groupby("City").size()
pd.concat([df1, df2])
pd.merge(df1, df2, on="id")
df1.join(df2)
df.to_csv("output.csv", index=False)
df.to_excel("output.xlsx")
df.to_json("output.json")
