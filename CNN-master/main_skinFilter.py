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


import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
# import imutils
import pprint
from matplotlib import pyplot as plt


def extractSkin(image):
    # Taking a copy of the image
    img = image.copy()
    # Converting from BGR Colours Space to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining HSV Threadholds
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

    # Cleaning up mask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    # Return the Skin image
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)


def removeBlack(estimator_labels, estimator_cluster):

    # Check for black
    hasBlack = False

    # Get the total number of occurance for each color
    occurance_counter = Counter(estimator_labels)

    # Quick lambda function to compare to lists
    def compare(x, y): return Counter(x) == Counter(y)

    # Loop through the most common occuring color
    for x in occurance_counter.most_common(len(estimator_cluster)):

        # Quick List comprehension to convert each of RBG Numbers to int
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]

        # Check if the color is [0,0,0] that if it is black
        if compare(color, [0, 0, 0]) == True:
            # delete the occurance
            del occurance_counter[x[0]]
            # remove the cluster
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break

    return (occurance_counter, estimator_cluster, hasBlack)


def getColorInformation(estimator_labels, estimator_cluster, hasThresholding=False):

    # Variable to keep count of the occurance of each color predicted
    occurance_counter = None

    # Output list variable to return
    colorInformation = []

    # Check for Black
    hasBlack = False

    # If a mask has be applied, remove th black
    if hasThresholding == True:

        (occurance, cluster, black) = removeBlack(
            estimator_labels, estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        hasBlack = black

    else:
        occurance_counter = Counter(estimator_labels)

    # Get the total sum of all the predicted occurances
    totalOccurance = sum(occurance_counter.values())

    # Loop through all the predicted colors
    for x in occurance_counter.most_common(len(estimator_cluster)):

        index = (int(x[0]))

        # Quick fix for index out of bound when there is no threshold
        index = (index-1) if ((hasThresholding & hasBlack)
                              & (int(index) != 0)) else index

        # Get the color number into a list
        color = estimator_cluster[index].tolist()

        # Get the percentage of each color
        color_percentage = (x[1]/totalOccurance)

        # make the dictionay of the information
        colorInfo = {"cluster_index": index, "color": color,
                     "color_percentage": color_percentage}

        # Add the dictionary to the list
        colorInformation.append(colorInfo)

    return colorInformation


def extractDominantColor(image, number_of_colors=5, hasThresholding=False):

    # Quick Fix Increase cluster counter to neglect the black(Read Article)
    if hasThresholding == True:
        number_of_colors += 1

    # Taking Copy of the image
    img = image.copy()

    # Convert Image into RGB Colours Space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape Image
    img = img.reshape((img.shape[0]*img.shape[1]), 3)

    # Initiate KMeans Object
    estimator = KMeans(n_clusters=number_of_colors, random_state=0)

    # Fit the image
    estimator.fit(img)

    # Get Colour Information
    colorInformation = getColorInformation(
        estimator.labels_, estimator.cluster_centers_, hasThresholding)
    return colorInformation


def plotColorBar(colorInformation):
    # Create a 500x100 black image
    color_bar = np.zeros((100, 500, 3), dtype="uint8")

    top_x = 0
    for x in colorInformation:
        bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

        color = tuple(map(int, (x['color'])))

        cv2.rectangle(color_bar, (int(top_x), 0),
                      (int(bottom_x), color_bar.shape[0]), color, -1)
        top_x = bottom_x
    return color_bar


N_EPOCHS = 20
BATCH_SIZE = 32

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
    train_images, test_images, train_labels, test_labels = train_test_split(image_names, labels, test_size=0.2)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25,
                                                                          random_state=1)
    # train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1)
    return train_images, test_images, train_labels, test_labels, val_images, val_labels

def findUniqueLabels(path="Dataset"):
    return os.listdir(path)

def createFolders(unique_labels, path='Data'):
    if not (os.path.exists(path)):
        os.makedirs(path)

    under_data_folder = ['train', 'test', 'validation']
    for folder_name in under_data_folder:
        if not (os.path.exists(os.path.join(path, folder_name))):
            os.makedirs(os.path.join(path, folder_name))

        for labels in unique_labels:
            if not (os.path.exists(os.path.join(path, folder_name, labels))):
                os.makedirs(os.path.join(path, folder_name, labels))

def prepareTrainAndTestFolder(path, prev_path, images, labels):
    #min_YCrCb = np.array([0, 133, 77], np.uint8)
    #max_YCrCb = np.array([235, 173, 127], np.uint8)

    for i in range(0, len(images)):
        img = cv2.imread(os.path.join(prev_path, labels[i], images[i]), 1)
        #imageYCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        #skinRegionYCrCb = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
        #newimage = cv2.bitwise_and(img, img, mask=skinRegionYCrCb)
        skin = extractSkin(img) #applied skin mask
        # dominantColors = extractDominantColor(skin, hasThresholding=True)



        resized_img = cv2.resize(skin, (200, 200), interpolation=cv2.INTER_AREA)
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

    validation_data = []
    for label in unique:
        class_num = unique.index(label)
        path = 'Data\\validation\\' + label
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), 1)
            validation_data.append([img_array, class_num])

    test_data = []
    for label in unique:
        class_num = unique.index(label)
        path = 'Data\\test\\' + label
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), 1)
            test_data.append([img_array, class_num])

    random.shuffle(train_data)
    random.shuffle(validation_data)
    random.shuffle(test_data)

    print("number of train images: ", len(train_data))
    print("number of validation images: ", len(validation_data))
    print("number of test images: ", len(test_data))

    return train_data, test_data, validation_data

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
def trainCNNModel(trainX, trainy, valX, valy):
    print("Creating model")

    if not (os.path.exists("checkpoints")):
        os.makedirs("checkpoints")

    checkpointer = ModelCheckpoint(filepath=os.path.join('checkpoints', "CNN_weights" + '.hdf5'), verbose=1)
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=trainX.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(trainX, trainy, validation_data=(valX, valy), batch_size=BATCH_SIZE, epochs=N_EPOCHS,callbacks=[checkpointer])
    return model

def testCNNModel(model, testX, testy):
    scores = model.evaluate(x=testX, y=testy, batch_size=BATCH_SIZE, verbose=1)
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
        train_images, test_images, train_labels, test_labels, val_images, val_labels = splitDatasetTrainAndTest(image_names, labels)
        createFolders(findUniqueLabels())
        prepareTrainAndTestFolder('Data\\train', 'Dataset', train_images, train_labels)
        prepareTrainAndTestFolder('Data\\test', 'Dataset', test_images, test_labels)
        prepareTrainAndTestFolder('Data\\validation', 'Dataset', val_images, val_labels)

    train, test, validation = createImageForTrainingAndTesting(findUniqueLabels())
    trainX, trainy = divideDataAndLabel(train)
    testX, testy = divideDataAndLabel(test)
    valX, valy = divideDataAndLabel(validation)
    model = trainCNNModel(trainX, trainy, valX, valy)
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
