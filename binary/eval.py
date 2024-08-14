import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

# Self-defined utils
from svm import SVMTrainer
from kernel import Kernel
from load_datasets import Dataset

step = 25
start = 4

def test_rbf(train_sample, train_label, test_sample, test_label, C, p, skl=False, debug=False, skl_type="svm", vis=False):
    multiple_run = False

    if skl:
        if skl_type == "svm":
            clf = svm.SVC(C=C)
        elif skl_type == "knn":
            clf = KNeighborsClassifier(n_neighbors=3)
        else:
            multiple_run = True
        if not multiple_run:
            clf.fit(train_sample, train_label)
            pred = lambda x: clf.predict([x])[0]
    else:
        num_features = len(train_sample[0])
        s = (num_features * train_sample.var() / 2) ** 0.5
        kernel = Kernel.gaussian(sigma=s)
        trainer = SVMTrainer(kernel, C, p)
        predictor = trainer.train(train_sample, train_label)
        pred = lambda x: predictor.predict(x)

    if not multiple_run:
        if debug:
            accuracy = np.mean([pred(x) == y for x, y in zip(test_sample, test_label)])
            if not skl:
                print(f"Test accuracy: {round(accuracy * 100, 2)}")
                print(f"nSV: {round(len(predictor._support_vectors) * 100 / len(train_sample), 1)} %")
                if vis:
                    return accuracy, np.mean([pred(x) == y for x, y in zip(train_sample, train_label)]), len(predictor._support_vectors) / len(train_sample)
            else:
                print(f"Test accuracy: {round(accuracy * 100, 2)} (Scikit-learn {skl_type})")
        else:
            return np.mean([pred(x) == y for x, y in zip(test_sample, test_label)])
    else:
        avg_acc = 0
        runs = 20
        for k in range(runs):
            if skl_type == "LDA":
                clf = LinearDiscriminantAnalysis()
            if skl_type == "random-forest":
                clf = RandomForestClassifier()
            if skl_type == "grad-boosting":
                clf = GradientBoostingClassifier()
            if skl_type == "decision-tree":
                clf = DecisionTreeClassifier()
            if skl_type == "adaboost":
                clf = AdaBoostClassifier(n_estimators=50, algorithm="SAMME")
            if skl_type == "gaussian-nb":
                clf = GaussianNB()
            clf.fit(train_sample, train_label)
            pred = lambda x: clf.predict([x])[0]
            avg_acc += np.mean([pred(x) == y for x, y in zip(test_sample, test_label)]) / runs
        print(f"Test accuracy: {round(avg_acc * 100, 2)} (avg, Scikit-learn {skl_type}, {runs} runs)")
        return avg_acc

# Evaluate the performance of a model
def eval(p, dataset, k_fold=5, test_size=0.3,
         split_seed=42, k_fold_seed=42, skl=False, skl_type="svm"):
    samples = dataset.samples
    labels = dataset.labels

    # train / test split based on random seed.
    train_sample, test_sample, train_label, test_label = train_test_split(
        samples, labels, test_size=test_size, random_state=split_seed
    )

    kf = KFold(n_splits=k_fold, shuffle=True, random_state=k_fold_seed)

    chosen_C = 1
    max_acc = 0

    if (not skl) or (skl_type == "svm"):
        for C in [0.1, 0.5, 1, 5, 10]:
            avg_acc = 0
            for train_index, test_index in kf.split(train_sample):
                cur_train_sample = train_sample[train_index]
                cur_train_label = train_label[train_index]
                cur_test_sample = train_sample[test_index]
                cur_test_label = train_label[test_index]
                acc = test_rbf(cur_train_sample, cur_train_label, cur_test_sample, cur_test_label, C, p, skl, skl_type=skl_type)
                avg_acc += acc / k_fold
            if avg_acc > max_acc:
                chosen_C = C
                max_acc = avg_acc
        print(f"Running on C = {chosen_C} from CV ...")

    test_rbf(train_sample, train_label, test_sample, test_label, chosen_C, p, skl, debug=True, skl_type=skl_type)

def eval_vis(p, C, dataset, N, test_size, split_seed=42):
    samples = dataset.samples
    labels = dataset.labels

    train_sample, test_sample, train_label, test_label = train_test_split(
        samples, labels, test_size=test_size, random_state=split_seed
    )

    test_acc_lst = []
    train_acc_lst = []
    nSV_lst = []

    for k in range(start, N + 1, step):
        cur_train_sample = train_sample[:k]
        cur_train_label = train_label[:k]
        test_acc, train_acc, nSV = test_rbf(cur_train_sample, cur_train_label, test_sample, test_label, C, p, debug=True, vis=True)
        print(f"k = {k}, test_acc = {test_acc}, train_acc = {train_acc}, nSV = {nSV}")
        test_acc_lst.append(test_acc)
        train_acc_lst.append(train_acc)
        nSV_lst.append(nSV)

    return test_acc_lst, train_acc_lst, nSV_lst

def comp(p, datasets):
    for dataset_name, test_size in datasets:
        dataset = Dataset(dataset_name)
        eval(p, dataset, test_size=test_size)
        eval(p, dataset, test_size=test_size, skl=True, skl_type="svm")
        eval(p, dataset, test_size=test_size, skl=True, skl_type="gaussian-nb")
        eval(p, dataset, test_size=test_size, skl=True, skl_type="random-forest")
        eval(p, dataset, test_size=test_size, skl=True, skl_type="knn")
        eval(p, dataset, test_size=test_size, skl=True, skl_type="grad-boosting")
        eval(p, dataset, test_size=test_size, skl=True, skl_type="LDA")
        eval(p, dataset, test_size=test_size, skl=True, skl_type="decision-tree")
        eval(p, dataset, test_size=test_size, skl=True, skl_type="adaboost")

p = 1.5
comp(p, [["breast_cancer", 0.3],
        ["heart", 0.3],
        ["ionosphere", 0.3],
        ["wine", 0.9],
        ["banknote", 0.7]])

def plot_fig(N, C, dataset_name, test_size):
    k_values = np.arange(start, N + 1, step)
    x_values = k_values / (((N - start) // step) * step + start)

    data = []
    p_list = [1.0, 10/8, 9/7, 8/6, 7/5, 6/4, 5/3, 4/2, 3.0]
    p_idx = 0

    for p in p_list:
        test_values, train_values, nSV_values = eval_vis(p, C, Dataset(dataset_name), N, test_size)
        data.append((test_values, train_values, nSV_values))

    fig, axs = plt.subplots(3, 3, figsize=(10, 5))

    for i, ax in enumerate(axs.flatten()):
        test_values, train_values, nSV_values = data[i]
        ax.plot(x_values, test_values, color="#8ECFC9", label='Test')
        ax.plot(x_values, train_values, color="#FA7F6F", label='Train')

        ax2 = ax.twinx()
        ax2.plot(x_values, nSV_values, color="#BEB8DC", label='nSV')
        ax.set_title(f'p = {round(p_list[p_idx], 2)}')
        ax.set_xlabel('Train size')
        ax.set_ylabel('Acc')
        ax2.set_ylabel('nSV')
        ax.set_ylim(0.55, 1.05)
        ax2.set_ylim(0.75, 1.05)
        p_idx += 1

        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()

        if i == 0:
            all_handles = handles + handles2
            all_labels = labels + labels2

    fig.legend(all_handles, all_labels, loc='center right', fontsize="medium")
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig("plot.pdf")
    plt.show()

plot_fig(649, 0.5, "wine", 0.9)
