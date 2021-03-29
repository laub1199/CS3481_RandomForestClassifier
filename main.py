from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    filepath = "./column_3C.dat"
    names = ["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle", "sacral_slope", "pelvic_radius",
             "degree_spondylolisthesis", "class"]

    df = pd.read_csv(filepath, sep=" ", header=None, names=names)

    x = df.drop('class', axis=1)
    y = df['class']

    train_data, text_data, train_label, test_label = train_test_split(x, y, test_size=0.2, random_state=6)

    list = []
    for i in range(1,20):
        clf = RandomForestClassifier(n_estimators=i)
        clf = clf.fit(train_data, train_label)
        prediction = clf.predict(text_data)
        accuracy = accuracy_score(test_label, prediction)
        print(str(i) + ': ' + str(accuracy))
        obj = [i, accuracy]
        list.append(obj)

    list = sorted(list, key=lambda x:x[0], reverse=True)

    print(list)