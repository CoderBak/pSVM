import svmpy
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

def eval(samples, labels, kernel, C, p):
    num_samples = len(labels)
    translate = np.vectorize(lambda x: 1.0 if x == 1 else -1.0)
    labels = translate(np.array(labels).reshape(num_samples, 1))
    trainer = svmpy.SVMTrainer(kernel, C, p)

    samples = scale(samples)

    train_sample, test_sample, train_label, test_label = train_test_split(
        samples, labels, test_size=0.2, random_state=1234
    )

    start_time = time.time()
    predictor = trainer.train(train_sample, train_label)
    end_time = time.time()

    #print(f"Number of features: {len(samples[0])}")
    #print(f"Train set size: {len(train_sample)}")
    #accuracy = np.mean([predictor.predict(x) == y for x, y in zip(train_sample, train_label)])
    #print(f"Train accuracy: {accuracy}")
    accuracy = np.mean([predictor.predict(x) == y for x, y in zip(test_sample, test_label)])
    print(f"Test accuracy: {round(accuracy * 100, 1)}")
    t = end_time - start_time
    if t >= 100:
        print(f"Time elapsed: {round(end_time - start_time, 0)} s")
    elif t >= 10:
        print(f"Time elapsed: {round(end_time - start_time, 1)} s")
    else:
        print(f"Time elapsed: {round(end_time - start_time, 2)} s")
    print(f"nSV: {round(len(predictor._support_vectors) * 100 / len(train_sample), 1)} %")

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_openml

cancer = load_breast_cancer()
spam = fetch_openml("spambase", version=1, as_frame=False)
diabete = fetch_openml("diabetes", version=1, as_frame=False)

s = 10
c = 1
p = 3

print("##############################################")
print("Testing on Breast Cancer Wisconsin dataset ...")
eval(cancer.data, cancer.target,
     svmpy.Kernel.gaussian(sigma=s), C=c, p=p)

print("##############################################")
print("Testing on Pima Indians Diabetes Dataset ...")
translate = np.vectorize(lambda x: 1 if x == "tested_positive" else 0)
eval(diabete.data, translate(diabete.target),
     svmpy.Kernel.gaussian(sigma=s), C=c, p=p)
