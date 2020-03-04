import numpy as np 
import matplotlib.pyplot as plt
import os 
import cv2
import random
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LinearRegression
DATADIR = "./Dataset"
# CATEGORIES = ['0','1','2','3','4','5','6','7','8','9']
CATEGORIES = ['0','1']
training_data = []
IMG_SIZE = 100

def show_image(img_array):
    plt.imshow(img_array,cmap="gray")
    plt.show()


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        label = int(category) #label (in digit case the category itself is label)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)

                training_data.append([img_array,label])
            except Exception as e: 
                pass
           

create_training_data()
random.shuffle(training_data)
X=[]
y=[]
for features, label in training_data:
    features = np.array(features).reshape(IMG_SIZE*IMG_SIZE,1)
    X.append(features)
    y.append(label)


X = np.array(X).reshape(len(training_data), IMG_SIZE*IMG_SIZE)
print(X.shape)

X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)

clf = LinearRegression()
clf.fit(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)
train_accuracy = clf.score(X_train, y_train)

print("TEST", test_accuracy)
print("TRAIN", train_accuracy)