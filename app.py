import numpy as np 
import matplotlib.pyplot as plt
import os 
import cv2
import random
from sklearn import datasets, svm, metrics
from sklearn.model_selection import cross_validate, train_test_split
import joblib
EXAMPLE = "./Examples"
DATADIR = "./Dataset"
CATEGORIES = ['0','1','2','3','4','5','6','7','8','9']
model_file_name = 'svm_classifier.sav'
training_data = []
def get_hog_descriptor():
    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.0
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    return hog

def show_image(img_array):
    plt.imshow(img_array,cmap="gray")
    plt.show()

def train():
    hog = get_hog_descriptor()
    for category in CATEGORIES:    
        path = os.path.join(DATADIR, category)
    
        label = int(category) #label (in digit case the category itself is label)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                if img_array.shape[0] != 3024:
                    img_array = hog.compute(img_array)
                    training_data.append([img_array,label])

            except Exception as e: 
                print('........ERROR ...............')
                print(str(e))
                pass
           
    random.shuffle(training_data)
    X=[]
    y=[]


    for features, label in training_data:
        X.append(features)
        y.append(label)


    X = np.array(X).reshape(len(training_data), features.shape[0])
    print(X.shape)

    X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)


    classifier = svm.SVC(gamma=0.001)
    classifier.fit(X_train,y_train)
    # classifier = joblib.load(model_file_name)
    print("TEST ACCURACY: ", metrics.accuracy_score(classifier.predict(X_test),y_test))
    print("TRAIN ACCURACY: ",metrics.accuracy_score(classifier.predict(X_train),y_train))
    joblib.dump(classifier, model_file_name)


def testing_examples(): 
    hog = get_hog_descriptor()
    path = os.path.join(EXAMPLE)
    loaded_model = joblib.load(model_file_name)

    for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                show_image(img_array)
                img_array = hog.compute(img_array)
                img_array = img_array.reshape(1,img_array.shape[0])
                y_val = loaded_model.predict(img_array)
                print("Label predicted:", y_val)
                

            except Exception as e: 
                print('........ERROR ...............')
                print(str(e))
                pass
    
    


train()