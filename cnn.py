import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
from tensorflow.python.framework import ops
import csv
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Conv2D,MaxPooling2D
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from tflearn.layers.conv import  conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from pickle import dump
from pickle import load
from sklearn.model_selection import train_test_split
import tflearn
from keras import applications
import keras as ks
from keras.utils import to_categorical

def start():
    TRAIN_DIR1 = 'Scenes training set/buildings'
    TRAIN_DIR2 = 'Scenes training set/forest'
    TRAIN_DIR3 = 'Scenes training set/glacier'
    TRAIN_DIR4 = 'Scenes training set/mountain'
    TRAIN_DIR5 = 'Scenes training set/sea'
    TRAIN_DIR6 = 'Scenes training set/street'
    TRAIN_DIR = []

    TRAIN_DIR.append(TRAIN_DIR1)
    TRAIN_DIR.append(TRAIN_DIR2)
    TRAIN_DIR.append(TRAIN_DIR3)
    TRAIN_DIR.append(TRAIN_DIR4)
    TRAIN_DIR.append(TRAIN_DIR5)
    TRAIN_DIR.append(TRAIN_DIR6)

    TEST_DIR = 'Scenes testing test'

    return TRAIN_DIR,TEST_DIR

def create_label(label_num):
    """ Create an one-hot encoded vector from label number """
    """
    # for kearas model
    if label_num == 0:
        return 0
    elif label_num == 1:
        return 1
    elif label_num == 2:
        return 2
    elif label_num == 3:
        return 3
    elif label_num == 4:
        return 4
    elif label_num == 5:
        return 5
    #for tflearn model
    """
    if label_num == 0:
        return np.array([1,0,0,0,0,0])
    elif label_num == 1:
        return np.array([0,1,0,0,0,0])
    elif label_num == 2:
        return np.array([0,0,1,0,0,0])
    elif label_num == 3:
        return np.array([0,0,0,1,0,0])
    elif label_num == 4:
        return np.array([0,0,0,0,1,0])
    elif label_num == 5:
        return np.array([0,0,0,0,0,1])

def create_train_data(TRAIN_DIR,IMG_SIZE):
    training_data = []
    for i in range(len(TRAIN_DIR)):
        for img in tqdm(os.listdir(TRAIN_DIR[i])):
            path = os.path.join(TRAIN_DIR[i], img)
            img_data = cv2.imread(path,cv2.IMREAD_COLOR)#cv2.IMREAD_COLOR
            img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
            img_data = cv2.cvtColor(img_data,cv2.COLOR_RGB2BGR)
            training_data.append([np.array(img_data), create_label(i)])

            #data_augmentation(img_data.reshape(-1, IMG_SIZE, IMG_SIZE, 3))
        #val = input("Enter your value: ")

    shuffle(training_data)
    dump(training_data, open('training_data.pkl', 'wb'))
    return training_data

def create_test_data(TEST_DIR,IMG_SIZE):
    testing_data=[]
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(path,cv2.IMREAD_COLOR)#
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        testing_data.append([np.array(img_data)])

    dump(testing_data, open('testing_data.pkl', 'wb'))
    return testing_data

def read(TRAIN_DIR,TEST_DIR,IMG_SIZE):
    if (os.path.exists('training_data.pkl')):  # If you have already created the dataset:
        training_data = load(open('training_data.pkl', 'rb'))
        # train_data = create_train_data()
    else:  # If dataset is not created:
        training_data = create_train_data(TRAIN_DIR,IMG_SIZE)

    if (os.path.exists('testing_data.pkl')):
        testing_data = load(open('testing_data.pkl', 'rb'))
        #print(np.shape(testing_data))
    else:
        testing_data = create_test_data(TEST_DIR,IMG_SIZE)

    train = training_data
    test = testing_data
    #print(np.shape(train))

    X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    data_augmentation(X_train)
    #print(X_train)
    #X_train=X_train/255.0
    y_train = [i[1] for i in train]

    return X_train,y_train,test


#################################################################

def transfer_learn(X_train,y_train,IMG_SIZE):
    #tune_model = Xception(weights='imagenet',include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    #vg=ks.applications.vgg16.VGG16()
    tune_model = applications.VGG16(weights='imagenet', include_top=False,input_shape=(IMG_SIZE, IMG_SIZE, 3))
    model = Sequential()

    for layer in tune_model.layers:
        model.add(layer)

    for layer in model.layers:
        layer.trainable = False


    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6,activation='softmax'))

    model.summary()

    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
    y_train1 = to_categorical(y_train1)
    y_test1 = to_categorical(y_test1)
    history=model.fit(X_train1, y_train1, epochs=25, validation_data = (X_test1, y_test1))

    plt.plot(history.history['val_loss'], 'r')

    #dump(model, open('vgg_model.pkl', 'wb'))
    """
    
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
    train_feature_ext = vgg_model.predict(X_train1)
    train_feature = train_feature_ext.reshape(train_feature_ext.shape[0], -1)

    dump(train_feature_ext, open('train_feature.pkl', 'wb'))
    # train_feature = load(open('train_feature.pkl', 'rb'))

    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)

    rf_model.fit(train_feature, y_train1)
    dump(rf_model, open('rf_model.pkl', 'wb'))
    # rf_model = load(open('rf_model.pkl', 'rb'))

    test_feature_ext = vgg_model.predict(X_test1)
    test_feature = test_feature_ext.reshape(test_feature_ext.shape[0], -1)
    dump(test_feature_ext, open('test_feature.pkl', 'wb'))

    pred_rf = rf_model.predict(test_feature)
    # le=preprocessing.LabelEncoder()
    # pred_rf=le.inverse_transform(pred_rf)
    print("acc = ", metrics.accuracy_score(y_test1, pred_rf))

    return rf_model,vgg_model
    """
    return model,history
########################################################

def data_augmentation(X_train):
    datagen_train = ImageDataGenerator(
        rotation_range=45,     #Random rotation between 0 and 45
        width_shift_range=0.2,   #% shift
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='constant', cval=125)

    # fit augmented image generator on data
    i = 0
    for batch in datagen_train.flow(X_train, batch_size=16,
                              save_to_dir='augmented',
                              save_prefix='',
                              save_format='jpg'):
        i += 1
        if i > 1:
            break  # otherwise the generator would loop indefinitely

    #print(np.shape(X_train))
    return X_train

#######################################################

def built_model(X_train,y_train,IMG_SIZE,MODEL_NAME,LR):
    ops.reset_default_graph()

    conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    conv1 = conv_2d(conv_input, 32, 5,regularizer='L1',weight_decay=0.0, activation='relu')
    pool1 = max_pool_2d(conv1, 5)

    conv2 = conv_2d(pool1, 64, 5, activation='relu')
    pool2 = max_pool_2d(conv2, 5)


    conv3 = conv_2d(pool2, 128, 5, activation='relu')
    pool3 = max_pool_2d(conv3, 5)


    conv4 = conv_2d(pool3, 64, 5, activation='relu')
    pool4 = max_pool_2d(conv4, 5)


    conv5 = conv_2d(pool4, 32, 5, activation='relu')
    pool5 = max_pool_2d(conv5, 5)


    fully_layer = fully_connected(pool5, 1024, activation='relu')
    fully_layer = dropout(fully_layer, 0.5)

    cnn_layers = fully_connected(fully_layer, 6, activation='softmax')
    cnn_layers = regression(cnn_layers, optimizer='Adam', learning_rate=LR, loss='categorical_crossentropy',
                            name='targets')
    model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)
    #model.load('rf_model.pkl',True)



    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

    if (os.path.exists('model.tfl.meta')):
        model.load('./model.tfl')

    else:
        early_stopping_cb = EarlyStoppingCallback(val_acc_thresh=0.95)
        #X_train1 = load(open('train_feature.pkl', 'rb'))
        #X_test1 = load(open('test_feature.pkl', 'rb'))

        model.fit({'input': X_train1}, {'targets': y_train1}, n_epoch=2,
                            validation_set=({'input': X_test1}, {'targets': y_test1}),
                            callbacks=early_stopping_cb,
                            snapshot_step=200, show_metric=True, run_id=MODEL_NAME)

        model.save('model.tfl')

        #accuracy_score = model.evaluate(x=X_test1, y=y_test1)["accuracy"]
        #print('Accuracy: {0:f}'.format(accuracy_score))

    return model

########################################################################

class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, val_acc_thresh):
        """ Note: We are free to define our init function however we please. """
        self.val_acc_thresh = val_acc_thresh

    def on_epoch_end(self, training_state):
        """ """
        # Apparently this can happen.
        if training_state.val_acc is None: return
        if training_state.val_acc > self.val_acc_thresh:
            raise StopIteration

########################################################

###plot
def plot(history):

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

#######################################################################
def test_res(test,model,IMG_SIZE,TEST_DIR):
    test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    #test = test / 255.0
    print(model)
    #test_feature_ext=vgg.predict(test)
    #test_feature=test_feature_ext.reshape(test_feature_ext.shape[0],-1)
    #pred_rf=model.predict(test_feature)
    # print(pred_rf)
    prediction = model.predict(test)

    header = ['Image', 'Label']
    arr = []
    c = 0

    with open('submit.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for img in tqdm(os.listdir(TEST_DIR)):
            word_label = img
            arr.append(word_label)

            index_max = np.argmax(prediction[c])

            if index_max == 0:
                arr.append('0')
            elif index_max == 1:
                arr.append('1')
            elif index_max == 2:
                arr.append('2')
            elif index_max == 3:
                arr.append('3')
            elif index_max == 4:
                arr.append('4')
            elif index_max == 5:
                arr.append('5')

            writer.writerow(arr)
            arr = []
            c = c + 1





