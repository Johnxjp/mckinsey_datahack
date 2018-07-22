import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from scipy.stats import boxcox

def premium_to_income(df, col_name="premium_income_ratio"):
    col = df["premium"] / df["Income"]
    col.name = col_name
    return col


def struggle_to_pay(df, col_name="struggling"):
    c1 = (df["Count_3-6_months_late"] > 1)
    c2 = (df["Count_6-12_months_late"] > 0)
    c3 = (df["Count_more_than_12_months_late"] > 0)
    c4 = (df["perc_premium_paid_by_cash_credit"] >= 0.8)
    condition = (c1 | c2 | c3) & c4

    df[col_name] = np.where(condition, True, False)
    return df


def late_renewal(df, col_name="late"):
    col = 1 * df["Count_3-6_months_late"] + 2 * df["Count_6-12_months_late"] + 3 * df["Count_more_than_12_months_late"]
    col.name = col_name
    return col


def label_encode(pd_series):
    le = LabelEncoder()
    return le.fit_transform(pd_series)


def split_data(df, test_size=0.2, random_state=None, split_train_test=True):
    y = df["renewal"]
    X = df.drop("renewal", axis=1)

    if split_train_test:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    else:
        return X, y


def perc_credit_quantise(df):
    df.loc[df["perc_premium_paid_by_cash_credit"] < 0.2, "perc_premium_paid_by_cash_credit"] = 0
    df.loc[(df["perc_premium_paid_by_cash_credit"] >= 0.2) & (df["perc_premium_paid_by_cash_credit"] < 0.8),
           "perc_premium_paid_by_cash_credit"] = 1
    df.loc[df["perc_premium_paid_by_cash_credit"] > 0.8, "perc_premium_paid_by_cash_credit"] = 2

    return df


def delayed_payment_ratio(df, col_name="delayed_payment_ratio"):
    col = (df["Count_3-6_months_late"] + df["Count_6-12_months_late"] + df["Count_more_than_12_months_late"]) \
          / df["no_of_premiums_paid"]
    col.name = col_name
    return col


def upsample_minority(df, n, random_state=1):
    """

    :param df:
    :param n: how many samples to draw
    :param random_state:
    :return:
    """
    # Separate majority and minority classes
    df_minority = df[df.renewal == 0]
    df_majority = df[df.renewal == 1]

    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=n,  # to match majority class
                                     random_state=random_state)  # reproducible results

    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    # Display new class counts
    print(df_upsampled.renewal.value_counts())

    return df_upsampled


def deal_with_na(df):
    df["application_underwriting_score"] = df["application_underwriting_score"].fillna(
        df["application_underwriting_score"].median())

    df["Count_3-6_months_late"] = df["Count_3-6_months_late"].fillna(0)
    df["Count_6-12_months_late"] = df["Count_6-12_months_late"].fillna(0)
    df["Count_more_than_12_months_late"] = df["Count_more_than_12_months_late"].fillna(0)

    return df


def loyalty_score(age, late_payments, on_time_payments, sourcing_channel):
    """
    creates scoring by quartile
    """

    # Get quartile information
    age_stats = age.describe()
    late_stats = late_payments.describe()
    on_time_payments_stats = on_time_payments.describe()

    # Age
    new_age = age.mask(age < age_stats["25%"], 1)
    new_age = new_age.mask((new_age >= age_stats["25%"]) & (new_age < age_stats["50%"]), 2)
    new_age = new_age.mask((new_age >= age_stats["50%"]) & (new_age < age_stats["75%"]), 3)
    new_age = new_age.mask(new_age >= age_stats["75%"], 4)

    # on-time payments
    new_on_time_payments = on_time_payments.mask(on_time_payments < on_time_payments_stats["25%"], 1)
    new_on_time_payments = new_on_time_payments.mask((new_on_time_payments >= on_time_payments_stats["25%"]) & (
                new_on_time_payments < on_time_payments_stats["50%"]), 2)
    new_on_time_payments = new_on_time_payments.mask((new_on_time_payments >= on_time_payments_stats["50%"]) & (
                new_on_time_payments < on_time_payments_stats["75%"]), 3)
    new_on_time_payments = new_on_time_payments.mask(new_on_time_payments >= on_time_payments_stats["75%"], 4)

    # late
    # new_late = late_payments.mask(late_payments < late_stats["25%"], 1)
    # new_late = new_late.mask((new_late >= late_stats["25%"]) & (new_late < late_stats["50%"]), 2)
    # new_late = new_late.mask((new_late >= late_stats["50%"]) & (new_late < late_stats["75%"]), 3)
    # new_late = new_late.mask(new_late >= late_stats["75%"], 4)

    # params
    a = 2
    # b = 0.25
    c = 5
    d = 1

    loyalty = a * new_age * new_on_time_payments + d * (sourcing_channel)

    return loyalty


def process_data(raw_data, upsample={"bool": True, "n": 10000}):
    processed_data = deal_with_na(raw_data)
    processed_data = processed_data.reset_index(drop=True)

    # drop id
    processed_data = processed_data.drop("id", axis=1)

    # processed_data = processed_data.drop("application_underwriting_score", axis=1)
    processed_data["residence_area_type"] = label_encode(processed_data["residence_area_type"])
    processed_data["sourcing_channel"] = label_encode(processed_data["sourcing_channel"])

    # upsample
    if upsample["bool"]:
        processed_data = upsample_minority(processed_data, n=upsample["n"])

    # sum late counts
    # processed_data["tot_lates"] = processed_data["Count_3-6_months_late"] + processed_data["Count_6-12_months_late"] + \
    #                               processed_data["Count_more_than_12_months_late"]

    # create new columns
    processed_data = pd.concat([premium_to_income(processed_data), processed_data], axis=1)
    processed_data = pd.concat([delayed_payment_ratio(processed_data), processed_data], axis=1)

    # Transform
    # processed_data["Income"] = np.log(processed_data["Income"])
    # processed_data["no_of_premiums_paid"], _ = boxcox(processed_data["no_of_premiums_paid"])
    # processed_data["application_underwriting_score"] = np.log(processed_data["application_underwriting_score"])
    # processed_data["age_in_days"], _ = boxcox(processed_data["age_in_days"])

    processed_data["loyalty"] = loyalty_score(processed_data["age_in_days"], processed_data["age_in_days"],
                                              processed_data["no_of_premiums_paid"], processed_data["sourcing_channel"])

    print("shape:", processed_data.shape)

    return processed_data


def binarise(col):
    ser = pd.Series(np.where(col > 0, 1, 0))
    ser.name = col.name
    return ser
