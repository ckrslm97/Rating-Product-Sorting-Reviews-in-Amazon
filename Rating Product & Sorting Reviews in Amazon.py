######################################################
#     Rating Product & Sorting Reviews in Amazon     #
######################################################
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
df = pd.read_csv("amazon_review.csv")

df.head()


average_rating =df.groupby("asin").agg({"overall":"mean"})

# Güncel Yorumlara göre Average Rating Hesaplama #

def time_based_weighted_rating(dataframe, w1 = 28 , w2 = 26, w3 = 24 , w4 = 22):
    return dataframe.loc[dataframe["day_diff"] <= 60, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 60) & (dataframe["day_diff"] <= 120), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 120) & (dataframe["day_diff"] <= 365), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 365), "overall"].mean() * w4 / 100

average_rating

time_based_weighted_rating(df)

df["helpful_yes"].value_counts()
df["helpful_yes"].max() # 1952
df["helpful_yes"].min() # 0

# Kullanıcının Yaptığı Yorumların Faydalı Bulunup Bulunmamasına Göre Average Rating Hesaplama #

def helpful_comment_based_rating(dataframe, w1 = 55, w2 = 25,w3 = 20):
    return dataframe.loc[dataframe["helpful_yes"] <= 100, "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["helpful_yes"] > 100) & (dataframe["helpful_yes"] <= 500), "overall"].mean() * w2 / 100 + \
           dataframe.loc[dataframe["helpful_yes"] < 2000, "overall"].mean() * w1 / 100


helpful_comment_based_rating(df)


"""
def avr_weighted_rating(dataframe, w1 = 60, w2 = 40):
    return (time_based_weighted_rating(dataframe) * w1 /100) + (helpful_comment_based_rating(dataframe) * w2 / 100)

avr_weighted_rating(df)
"""

## GÖREV - 2 ##

# Ürün detay sayfasında görüntülenecek 20 reviewi belirleme #

def wilson_lower_bound(up , down, confidence=0.95):
    n = up + down

    if n == 0:
        return 0

    z = st.norm.ppf(1 - (1 - confidence) / 2)

    phat = 1.0 * up / n
    return (phat + z * z / (2*n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["helpful_not"] = df['total_vote'] - df['helpful_yes']

df['wilson_lower_bound_score']  = df.apply(lambda x: wilson_lower_bound(x['helpful_yes'], x['helpful_not']) , axis=1)

df.sort_values('wilson_lower_bound_score', ascending=False).head(20)
