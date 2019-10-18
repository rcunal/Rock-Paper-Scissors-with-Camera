from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
import keras
import cv2
import os

def getImageNamesAndClassLabels(path='Dataset'):
    class_list = os.listdir(path)

    image_names = []
    labels = []
    for class_name in class_list:
        image_list = os.listdir(os.path.join(path, class_name))

        for image_name in image_list:
            image_names.append(image_name)
            labels.append(class_name)
    return image_names, labels

def splitDatasetTrainAndTest(image_names, labels):
    print("\nThe images are divided into test and train folder")
    train_images, test_images, train_labels, test_labels = train_test_split(image_names, labels, test_size=0.1)
    return train_images, test_images, train_labels, test_labels

def findUniqueLabels(path="Dataset"):
    return os.listdir(path)

def createFolders(unique_labels, path='Data'):
    if not (os.path.exists(path)):
        os.makedirs(path)

    under_data_folder = ['train', 'test']
    for folder_name in under_data_folder:
        if not (os.path.exists(os.path.join(path, folder_name))):
            os.makedirs(os.path.join(path, folder_name))

        for labels in unique_labels:
            if not (os.path.exists(os.path.join(path, folder_name, labels))):
                os.makedirs(os.path.join(path, folder_name, labels))

def prepareTrainAndTestFolder(path, prev_path, images, labels):
    for i in range(0, len(images)):
        img = cv2.imread(os.path.join(prev_path, labels[i], images[i]), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
        if not (os.path.isfile(os.path.join(path, labels[i], images[i]))):
            cv2.imwrite(os.path.join(path, labels[i], images[i]), resized_img)

def createImageForTrainingAndTesting(unique):
    print("The images and labels getting for training and testing\n")

    train_data = []
    for label in unique:
        class_num = unique.index(label)
        path = 'Data\\train\\' + label
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), 1)
            train_data.append([img_array, class_num])

    test_data = []
    for label in unique:
        class_num = unique.index(label)
        path = 'Data\\test\\' + label
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), 1)
            test_data.append([img_array, class_num])

    random.shuffle(train_data)
    random.shuffle(test_data)
    return train_data, test_data

def divideDataAndLabel(data_list):
    X = []
    y = []
    for features, label in data_list:
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, 200, 200, 3)
    X = X / 255.0
    y = keras.utils.to_categorical(y, 3)
    return X, y

""""Modeller kaydediliyor"""
def trainCNNModel(trainX, trainy):
    print("Creating model")

    if not (os.path.exists("checkpoints")):
        os.makedirs("checkpoints")

    checkpointer = ModelCheckpoint(filepath=os.path.join('checkpoints', "CNN_weights" + '.hdf5'), verbose=1)
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same', input_shape=trainX.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(trainX, trainy, batch_size=4, epochs=50,callbacks=[checkpointer])
    return model

def testCNNModel(model, testX, testy):
    scores = model.evaluate(x=testX, y=testy, batch_size=4, verbose=1)
    print("loss :" + str(scores[0]))
    print("acc  :" + str(scores[1]))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def predictImages(path, c_name, i_name):
    img_array = cv2.imread(os.path.join(path, c_name, i_name), 1)
    X = np.array(img_array).reshape(-1, 200, 200, 3)
    X = X / 255.0

    prediction = model.predict_classes(X)
    class_list = findUniqueLabels(path)
    for i, class_name in enumerate(class_list):
        if i == prediction:
            return class_name

def putText(image, label, c_name, i_name):
    if not (os.path.exists("Output")):
        os.makedirs("Output")

    cv2.putText(image, label, (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite("Output/" + c_name + '_' + i_name, image)

def draw_confusion_matrix(testX,testy):
    predictions = model.predict(testX)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = []
    for item in testy:
        for i, a in enumerate(item):
            if (a == 1):
                true_classes.append(i)
    cm = confusion_matrix(true_classes, predicted_classes)
    cm_plot_labels = ['rock', 'paper', 'scissors']
    plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

if __name__ == '__main__':
    if not (os.path.exists('Data')):
        image_names, labels = getImageNamesAndClassLabels()
        train_images, test_images, train_labels, test_labels = splitDatasetTrainAndTest(image_names, labels)
        createFolders(findUniqueLabels())
        prepareTrainAndTestFolder('Data\\train', 'Dataset', train_images, train_labels)
        prepareTrainAndTestFolder('Data\\test', 'Dataset', test_images, test_labels)

    train, test = createImageForTrainingAndTesting(findUniqueLabels())
    trainX, trainy = divideDataAndLabel(train)
    testX, testy = divideDataAndLabel(test)
    model = trainCNNModel(trainX, trainy)
    testCNNModel(model, testX, testy)
    draw_confusion_matrix(testX, testy)

    #model = load_model("checkpoints/CNN_weights.hdf5")

    print("\nwriting predicted labels on images")
    # test folder
    path = "Data\\test"
    wanted_class_names = ['rock', 'paper', 'scissors'] # you can add another labels
    class_names = os.listdir(path)
    for c_name in class_names:
        if c_name in wanted_class_names:
            image_names = os.listdir(os.path.join(path, c_name))
            for i_name in image_names:
                image_predict_class = predictImages(path, c_name, i_name)
                putText(cv2.imread(os.path.join(path, c_name, i_name), 1), image_predict_class, c_name, i_name)
