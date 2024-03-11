import face_recognition
import pickle
import cv2
import os
from tqdm import tqdm
import numpy as np

from keras.models import load_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, Input, Concatenate, Layer
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import TensorBoard  #, ModelCheckpoint, EarlyStopping
from keras.models import load_model
from tensorflow.keras.applications import ResNet50, VGG16, MobileNet
from sklearn.metrics import confusion_matrix

class FaceData:
    def __init__(self, imgPath = '', fileName = '', isNew = True):
        self.imgPath = imgPath
        if (fileName == ''):
            self.fileName = imgPath + 'Data.txt'
        else:
            self.fileName = fileNames
        self.dim = (120, 120)
        self.isNew = isNew
        self.imgPathTemp = imgPath + 'Crop'

    def encode(self):
        print("Encode Images: ")
        personNum, data = self.parseFolder()
        encodingFaces = [['Name', 'FaceEncode']]
        for i, row in enumerate(tqdm(data)):
            image = cv2.cvtColor(cv2.imread(row[1]), cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(image, model='hog')
            encoding = face_recognition.face_encodings(image, boxes)
            encodingFaces.append([row[0], encoding])
        print("End Encode.\n")


    def parseFolder(self):
        folder = self.imgPath if self.isNew else self.imgPathTemp
        folderPaths = os.listdir(folder)
        photoWithName = []
        print("Read Images: ")
        for i, path in enumerate(tqdm(folderPaths)):
            imagePaths = os.listdir(folder + '\\' + path)
            for j, imagePath in enumerate(imagePaths):
                filePath = folder + '\\' + path + '\\' + imagePath
                if os.path.isfile(filePath):
                    photoWithName.append([path, filePath])
        print("End Read.\n")
        return len(folderPaths), folderPaths, photoWithName

    def detectFace(self, name, imagePath):
        images = []
        image = cv2.imread(imagePath)
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        haarCascade = cv2.CascadeClassifier('Haarcascade_frontalface_default.xml')
        facesRect = haarCascade.detectMultiScale(grayImage, 1.1, 9)

        for (x, y, w, h) in facesRect:
            face = image[y:y+h, x:x+w]
            faceDim = cv2.resize(face, self.dim, interpolation = cv2.INTER_AREA)
            images.append(faceDim)

        return name, self.dim, images

    def detectTrainFace(self, name, imagePath):
        images = []
        image = cv2.imread(imagePath)
        imgFolder = self.imgPathTemp + '\\' + name
        fileName = os.path.basename(imagePath)
        imagePathNew = imgFolder + '\\' + fileName

        if self.isNew:
            grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            haarCascade = cv2.CascadeClassifier('Haarcascade_frontalface_default.xml')
            facesRect = haarCascade.detectMultiScale(grayImage, 1.1, 9)

            for (x, y, w, h) in facesRect:
                face = image[y:y+h, x:x+w]
                faceDim = cv2.resize(face, self.dim, interpolation = cv2.INTER_AREA)
                images.append(faceDim)

            if not os.path.exists(self.imgPathTemp):
                os.makedirs(self.imgPathTemp)
            if not os.path.exists(imgFolder):
                os.makedirs(imgFolder)
            if not os.path.exists(imagePathNew):
                try:
                    cv2.imwrite(imagePathNew, images[0])
                except:
                    print("Failed image {}, {}".format(name, imagePath))
        else:
            images = [cv2.resize(image, self.dim, interpolation = cv2.INTER_AREA)]

        return name, self.dim, images

    def trainData(self, dim = (120, 120)):
        self.dim = dim
        personNum, persons, data = self.parseFolder()

        print("Create Image Data: ")
        x = []
        y = []
        yDict = {}
        for i, image in enumerate(tqdm(data)):
            name, dim, images = self.detectTrainFace(image[0], image[1])

            personIndex = persons.index(name)
            try:
                x.append(images[0] / 255.0)
                y.append(personIndex)
                yDict[name] = personIndex
            except:
                print("Failed Image Data {}, {}".format(image[0], image[1]))

        y = to_categorical(np.array(y), personNum)
        personCat = to_categorical([i for i in range(personNum)], personNum)

        for i, name in enumerate(persons):
            yDict[name] = personCat[i]
        self.isNew = False
        print("End Image Data\n ")
        return x, y, dim, yDict

class NeuralNetwork:
    def __init__(self, input):
        self.input = input
        self.model = Model()
        self.isStart = True

    def addDense(self, dense):
        if self.isStart:
            self.model = dense(self.input)
            self.isStart = False
        else:
            self.model = dense(self.model)

class NetworkEnsemble:
    def __init__(self, neuron, act, xTrain, yTrain, numClasses, dim):
        self.networks = []
        self.model = Model()
        self.neuron = neuron
        self.act = act
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.lenNet = 0
        self.numClasses = numClasses
        self.dim = dim
        self.callbacks = []

    def addNetwork(self, networks):
        if type(networks)==list:
            for net in networks:
                self.networks.append(net)
        else:
            self.networks.append(networks)

        self.lenNet = len(self.networks)

    def createModel(self, isUnique = False):
        if (len(self.networks) > 1):
            out = Concatenate()([net.model for net in self.networks])
            out = Dense(self.neuron * 2, activation = self.act)(out)
            out = Dropout(0.25)(out)

            out = Dense(self.neuron // 2, activation = self.act)(out)
            out = Dropout(0.25)(out)

            out = Dense(self.neuron // 4, activation = self.act)(out)
            out = Dense(self.numClasses, activation = 'softmax')(out)

            self.model = Model([net.input for net in self.networks], out)

        else:
            out = self.networks[0].model
            if isUnique:
                self.model = Model(self.networks[0].input, out)
            else:
                out = Dense(self.neuron * 2, activation = self.act)(out)
                out = Dropout(0.25)(out)

                out = Dense(self.neuron // 2, activation = self.act)(out)
                out = Dropout(0.25)(out)

                out = Dense(self.neuron // 4, activation = self.act)(out)
                out = Dense(self.numClasses, activation = 'softmax')(out)

                self.model = Model([net.input for net in self.networks], out)

    def trainModel(self, loss, optimizer, metrics, validation_split, batch_size, epochs):
        self.model.summary()

        self.model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)

        print("\nTrain Model:")
        self.model.fit([np.array(self.xTrain) for net in self.networks],
                        np.array(self.yTrain),
                        validation_split = validation_split,
                        batch_size = batch_size,
                        epochs = epochs,
                        callbacks = self.callbacks)

    def testModel(self, xTest, yTest, batch_size, yDict = {}):
        print("\nTest Model:")
        self.model.evaluate([np.array(xTest) for net in range(self.lenNet)],
                             np.array(yTest),
                             batch_size=batch_size)
        self.testModelWithMatrix(xTest, yTest, yDict)

    def testModelWithMatrix(self, xTest, yTest, yDict = {}):
        yPred = []
        for xT in xTest:
            pred, name, val= self.predict(xT, yDict)
            yPred.append(val)
        matrix = confusion_matrix(np.array(yTest).argmax(axis=1), np.array(yPred).argmax(axis=1))
        print("Матрица: ")
        for row in matrix:
            print(row)
        print("-----------------------------")

    def saveModel(self, name, index = 0):
        fname = '{}_{}'.format(name, index)
        fileName = "{}.h5".format(fname)
        print(fileName)
        netFileName = "{}.txt".format(fname)
        try:
            if os.path.isfile(fileName):
                self.saveModel(name, index + 1)
            else:
                self.model.save(fileName)
                f = open(netFileName, 'w')
                f.write("{}".format(self.lenNet))
                f.close()
        except:
            pass

    def loadModel(self, name):
        f = open("{}.txt".format(name), 'r')
        self.model = load_model("{}.h5".format(name))
        for line in f:
            self.lenNet = int(line)
        f.close()

    def predict(self, image, dict = {}):
        (w, h) = self.dim
        if (self.lenNet == 1):
            predict = self.model.predict(np.array(image).reshape(1, w, h, 3))
        else:
            predict = self.model.predict([np.array(image).reshape(1, w, h, 3) for i in range(self.lenNet)])
        index = list(predict[0]).index(np.max(predict[0]))
        value = [1 if i == index else 0 for i in range(len(dict))]
        name = ""
        for i, el in dict.items():
            indexDict = np.where(el == np.max(el))
            if indexDict[0] == index:
                name = i
        return predict, name, value

    def addCallback(self, callbacks):
        if type(callbacks)==list:
            for callback in callbacks:
                self.callbacks.append(callback)
        else:
            self.callbacks.append(callbacks)


faceTrain = FaceData('train', isNew = False)
faceTest = FaceData('test', isNew = False)

xTrain, yTrain, dim, yTrainDict = faceTrain.trainData((224, 224))
xTest, yTest, dimTest, yTestDict = faceTest.trainData((224, 224))

_, _, images = faceTrain.detectFace("", "Alia Bhatt_9.jpg")
predictedFace = images[0]

numClasses = len(yTrainDict)
print(dim)
w, h = dim

inputShape = (w, h, 3)
act = 'relu'
neuron = 512
act2 = 'relu'
batchSize = 16
epochs = 25
validationSplit = 0
#
# tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
#                           write_graph=True, write_images=False)

net = NetworkEnsemble(neuron, act2, xTrain, yTrain, numClasses, dim)
net.loadModel("netResNet_2")
# net.testModel(xTest, yTest, 16, yTrainDict)
predict, name, val = net.predict(images[0], yTrainDict)
print(f"ResNet: {name}")
net.loadModel("netVGG_2")
# net.testModel(xTest, yTest, 16, yTrainDict)
predict, name, val = net.predict(images[0], yTrainDict)
print(f"VGG: {name}")
net.loadModel("netMobileNet_1")
# net.testModel(xTest, yTest, 16, yTrainDict)
predict, name, val = net.predict(images[0], yTrainDict)
print(f"MobileNet: {name}")
net.loadModel("netResNetVGG_1")
# net.testModel(xTest, yTest, 16, yTrainDict)
predict, name, val = net.predict(images[0], yTrainDict)
print(f"ResNetVGG: {name}")
net.loadModel("netResNetMobileNet_1")
# net.testModel(xTest, yTest, 16, yTrainDict)
predict, name, val = net.predict(images[0], yTrainDict)
print(f"ResNetMobileNet: {name}")
net.loadModel("netVGGMobileNet_1")
# net.testModel(xTest, yTest, 16, yTrainDict)
predict, name, val = net.predict(images[0], yTrainDict)
print(f"VGGMobileNet: {name}")
net.loadModel("netAll_1")
# net.testModel(xTest, yTest, 16, yTrainDict)
predict, name, val = net.predict(images[0], yTrainDict)
print(f"All: {name}")

# modelResNet = NeuralNetwork(Input(inputShape))
# modelResNet.addDense(ResNet50(include_top=True, input_shape=inputShape, weights = None, classes = numClasses))
#
# partResNet = NeuralNetwork(Input(inputShape))
# partResNet.addDense(ResNet50(include_top=False, input_shape=inputShape, weights = None))
# partResNet.addDense(Flatten())
#
# modelVGG = NeuralNetwork(Input(inputShape))
# modelVGG.addDense(VGG16(include_top=True, input_shape=inputShape, weights=None, classes = numClasses))
#
# partVGG = NeuralNetwork(Input(inputShape))
# partVGG.addDense(VGG16(include_top=False, input_shape=inputShape, weights=None))
# partVGG.addDense(Flatten())
#
# modelMobileNet = NeuralNetwork(Input(inputShape))
# modelMobileNet.addDense(MobileNet(include_top=True, input_shape=inputShape, weights=None, classes = numClasses))
#
# partMobileNet = NeuralNetwork(Input(inputShape))
# partMobileNet.addDense(MobileNet(include_top=False, input_shape=inputShape, weights=None))
# partMobileNet.addDense(Flatten())
#
# # netResNet = NetworkEnsemble(neuron, act2, xTrain, yTrain, numClasses, dim)
# # netResNet.addNetwork(modelResNet)
# # # netResNet.addCallback(tensorboard)
# # netResNet.createModel(True)
# # netResNet.trainModel('categorical_crossentropy', 'SGD', ['accuracy'], validationSplit, batchSize, epochs)
# # netResNet.saveModel('netResNet')
# # netResNet.testModel(xTest, yTest, 16, yTrainDict)
# #
# # netVGG = NetworkEnsemble(neuron, act2, xTrain, yTrain, numClasses, dim)
# # netVGG.addNetwork(modelVGG)
# # # netVGG.addCallback(tensorboard)
# # netVGG.createModel(True)
# # netVGG.trainModel('categorical_crossentropy', 'SGD', ['accuracy'], validationSplit, batchSize, epochs)
# # netVGG.saveModel('netVGG')
# # netVGG.testModel(xTest, yTest, 16, yTrainDict)
# #
# # netMobileNet = NetworkEnsemble(neuron, act2, xTrain, yTrain, numClasses, dim)
# # netMobileNet.addNetwork(modelMobileNet)
# # # netMobileNet.addCallback(tensorboard)
# # netMobileNet.createModel(True)
# # netMobileNet.trainModel('categorical_crossentropy', 'SGD', ['accuracy'], validationSplit, batchSize, epochs)
# # netMobileNet.saveModel('netMobileNet')
# # netMobileNet.testModel(xTest, yTest, 16, yTrainDict)
# #
# # netResNetVGG = NetworkEnsemble(neuron, act2, xTrain, yTrain, numClasses, dim)
# # netResNetVGG.addNetwork([partResNet, partVGG])
# # # netResNetVGG.addCallback(tensorboard)
# # netResNetVGG.createModel()
# # netResNetVGG.trainModel('categorical_crossentropy', 'SGD', ['accuracy'], validationSplit, batchSize, epochs)
# # netResNetVGG.saveModel('netResNetVGG')
# # netResNetVGG.testModel(xTest, yTest, 16, yTrainDict)
# #
# # netResNetMobileNet = NetworkEnsemble(neuron, act2, xTrain, yTrain, numClasses, dim)
# # netResNetMobileNet.addNetwork([partResNet, partMobileNet])
# # # netResNetMobileNet.addCallback(tensorboard)
# # netResNetMobileNet.createModel()
# # netResNetMobileNet.trainModel('categorical_crossentropy', 'SGD', ['accuracy'], validationSplit, batchSize, epochs)
# # netResNetMobileNet.saveModel('netResNetMobileNet')
# # netResNetMobileNet.testModel(xTest, yTest, 16, yTrainDict)
#
# netVGGMobileNet = NetworkEnsemble(neuron, act2, xTrain, yTrain, numClasses, dim)
# netVGGMobileNet.addNetwork([partVGG, partMobileNet])
# # netVGGMobileNet.addCallback(tensorboard)
# netVGGMobileNet.createModel()
# netVGGMobileNet.trainModel('categorical_crossentropy', 'SGD', ['accuracy'], validationSplit, batchSize, epochs)
# netVGGMobileNet.saveModel('netVGGMobileNet')
# netVGGMobileNet.testModel(xTest, yTest, 16, yTrainDict)
#
#
# netAll = NetworkEnsemble(neuron, act2, xTrain, yTrain, numClasses, dim)
# netAll.addNetwork([partResNet, partVGG, partMobileNet])
# # netAll.addCallback(tensorboard)
# netAll.createModel()
# netAll.trainModel('categorical_crossentropy', 'SGD', ['accuracy'], validationSplit, batchSize, epochs)
# netAll.saveModel('netAll')
# netAll.testModel(xTest, yTest, 16, yTrainDict)
#
# # print("VGG: \n")
# # predict, value, _ = netVGG.predict(predictedFace, yTrainDict)
# # print(f"\nPredict {predict}. \nValue: {value} \n")
# # print("-------- \n")
# #
# # print("ResNet: \n")
# # predict, value, _ = netResNet.predict(predictedFace, yTrainDict)
# # print(f"\nPredict {predict}. \nValue: {value} \n")
# # print("-------- \n")
# #
# # print("MobileNet: \n")
# # predict, value, _ = netMobileNet.predict(predictedFace, yTrainDict)
# # print(f"\nPredict {predict}. \nValue: {value} \n")
# # print("-------- \n")
# #
# # print("ResNetVGG: \n")
# # predict, value, _ = netResNetVGG.predict(predictedFace, yTrainDict)
# # print(f"\nPredict {predict}. \nValue: {value} \n")
# # print("-------- \n")
# #
# # print("ResNetMobileNet: \n")
# # predict, value, _ = netResNetMobileNet.predict(predictedFace, yTrainDict)
# # print(f"\nPredict {predict}. \nValue: {value} \n")
# # print("-------- \n")
# #
# print("VGGMobileNet: \n")
# predict, value, _ = netVGGMobileNet.predict(predictedFace, yTrainDict)
# print(f"\nPredict {predict}. \nValue: {value} \n")
# print("-------- \n")
#
# print("All: \n")
# predict, value, _ = netAll.predict(predictedFace, yTrainDict)
# print(f"\nPredict {predict}. \nValue: {value} \n")
# print("-------- \n")
#
#
# print('\n{}\n'.format(yTrainDict))
# print(model.predict(np.array(images[0]).reshape(1, 120, 120, 3)))
