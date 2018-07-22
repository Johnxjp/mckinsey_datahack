from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import mode

from data_utils import *


if __name__ == "__main__":

    print("loading raw data")
    raw_data = pd.read_csv("train_ZoGVYWq.csv")

    print("splitting into train and test")
    x_train, x_test, y_train, y_test = split_data(raw_data, test_size=0.2, random_state=1)

    print("processing data")
    processed_data = process_data(raw_data=pd.concat([x_train, y_train], axis=1), upsample={"bool": True, "n": 20000})
    test_data = process_data(raw_data=pd.concat([x_test, y_test], axis=1), upsample={"bool": False, "n": 20000})

    # processed_data["Count_3-6_months_late"] = binarise(processed_data["Count_3-6_months_late"])
    # processed_data["Count_6-12_months_late"] = binarise(processed_data["Count_6-12_months_late"])
    # processed_data["Count_more_than_12_months_late"] = binarise(processed_data["Count_more_than_12_months_late"])
    #
    # test_data["Count_3-6_months_late"] = binarise(test_data["Count_3-6_months_late"])
    # test_data["Count_6-12_months_late"] = binarise(test_data["Count_6-12_months_late"])
    # test_data["Count_more_than_12_months_late"] = binarise(test_data["Count_more_than_12_months_late"])

    y = processed_data['renewal']
    X = processed_data.drop("renewal", axis=1)
    # rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=100, max_depth=10)
    # rf.fit(X, y)

    y_test = test_data['renewal']
    x_test = test_data.drop("renewal", axis=1)
    # probs = rf.predict_proba(x_test)
    # roc = roc_auc_score(y_true=y_test, y_score=probs[:, 1])
    # roc

    iterations = 10
    preds = np.zeros(shape=(iterations, y_test.shape[0]))

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression



    for i in range(iterations):
        print("iteration", i)

        if i % 2 == 0:
            model = XGBClassifier(max_depth=np.random.randint(3, 10), n_estimators=np.random.randint(100, 150))
        elif i < 5:
            model = RandomForestClassifier(n_estimators=np.random.randint(100, 150), min_samples_leaf=np.random.randint(30, 50))
        else:
            model = RandomForestClassifier(n_estimators=100, min_samples_leaf=4)
        # else:
        # model = RandomForestClassifier(n_estimators=100, min_samples_leaf=40)
        # model = LogisticRegression()
        model.fit(X, y)
        probs = model.predict_proba(x_test)
        preds[i, :] = probs[:, 1]


    new_probs = mode(preds)[0]
    print(new_probs, new_probs.shape, y_test.shape)

    roc = roc_auc_score(y_true=y_test, y_score=new_probs.squeeze())

    print("roc", roc)

    # print(model.feature_importances_)
