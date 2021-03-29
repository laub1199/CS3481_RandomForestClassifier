from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import graphviz

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    filepath = "./column_3C.dat"
    names = ["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle", "sacral_slope", "pelvic_radius",
             "degree_spondylolisthesis", "class"]

    df = pd.read_csv(filepath, sep=" ", header=None, names=names)

    x = df.drop('class', axis=1)
    y = df['class']

    train_data, test_data, train_label, test_label = train_test_split(x, y, test_size=0.2)

    train_list = []
    test_list = []
    clfs = []
    for i in range(1,20):
        clf = RandomForestClassifier(n_estimators=i)
        clf = clf.fit(train_data, train_label)

        # train_prediction = clf.predict(train_data)
        # train_accuracy = accuracy_score(train_label, train_prediction)

        test_prediction = clf.predict(test_data)
        clfs.append(clf)
        test_accuracy = accuracy_score(test_label, test_prediction)

        # print(str(i) + ': ' + str(train_accuracy) + str(test_accuracy))
        print(str(i) + ': ' + str(test_accuracy))

        # obj = [i, train_accuracy]
        # train_list.append(obj)

        obj = [i, test_accuracy]
        test_list.append(obj)

    # train_list = sorted(train_list, key=lambda x:x[1], reverse=True)
    test_list = sorted(test_list, key=lambda x:x[1], reverse=True)

    # print(train_list)
    print(test_list)


    clf_loc = test_list[0][0]
    num_of_estimator = clf_loc + 1

    print(str(clf_loc) + ': ')

    print(clfs[clf_loc])

    for i in range (0, num_of_estimator):
        dot_data = tree.export_graphviz(clfs[clf_loc].estimators_[i], out_file=None, filled=True,
                                        feature_names=["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle",
                                                       "sacral_slope", "pelvic_radius", "degree_spondylolisthesis"],
                                        class_names=["Hernia", "Normal", "Spondylolisthesis"], rounded=True,
                                        special_characters=True)

        graph_training = graphviz.Source(dot_data)
        graph_training.render('Graph' + str(i), view=True)