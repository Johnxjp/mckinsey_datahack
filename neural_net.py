from keras.layers import Dense, Input, Dropout
from keras.models import Sequential
from keras.optimizers import SGD

from sklearn.preprocessing import StandardScaler

from xgboost_tree import *
from data_utils import *


def add_channels(df):
    df["channel_a"] = df["sourcing_channel"].apply(lambda x: 1 if x==0 else 0)
    df["channel_b"] = df["sourcing_channel"].apply(lambda x: 1 if x == 1 else 0)
    df["channel_c"] = df["sourcing_channel"].apply(lambda x: 1 if x == 2 else 0)
    df["channel_d"] = df["sourcing_channel"].apply(lambda x: 1 if x == 3 else 0)
    df["channel_e"] = df["sourcing_channel"].apply(lambda x: 1 if x == 4 else 0)

    df = df.drop("sourcing_channel", axis=1)
    return df


def normalise_col(df, columns):

    scaler = StandardScaler(copy=True)
    for col in columns:
        print(df[col].values.shape)
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

    return df


def define_model(input_shape):
    model = Sequential()
    model.add(Dense(32, activation="sigmoid", input_shape=(input_shape,)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="sigmoid"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adadelta", loss="binary_crossentropy", metrics=["acc"])
    return model


if __name__ == "__main__":
    print("loading raw data")
    raw_data = pd.read_csv("train_ZoGVYWq.csv")

    print("splitting into train and test")
    x_train, x_test, y_train, y_test = split_data(raw_data, test_size=0.2, random_state=1)

    processed_data = process_data(raw_data=pd.concat([x_train, y_train], axis=1), upsample={"bool": True, "n": 10000})
    test_data = process_data(raw_data=pd.concat([x_test, y_test], axis=1), upsample={"bool": False, "n": 20000})

    processed_data = add_channels(processed_data)
    test_data = add_channels(test_data)

    cols = ['delayed_payment_ratio',
            'premium_income_ratio',
            'perc_premium_paid_by_cash_credit',
            'age_in_days',
            'Income',
            'application_underwriting_score',
            'no_of_premiums_paid',
            'premium',
            'loyalty'
            ]

    processed_data = normalise_col(processed_data, cols)
    test_data = normalise_col(test_data, cols)

    x_train, y_train = split_data(processed_data, split_train_test=False)
    x_test, y_test = split_data(test_data, split_train_test=False)

    processed_data_numpy = processed_data.values

    model = define_model(input_shape=x_train.shape[1])

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=1)
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=64, epochs=10)

    preds = model.predict_proba(x_test)
    roc = roc_auc_score(y_test, preds)
    print("roc", roc)