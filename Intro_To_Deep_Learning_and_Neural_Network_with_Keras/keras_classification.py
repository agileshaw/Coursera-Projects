from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras

def classification(n_pixels, n_classes):
    model = Sequential()
    model.add(Dense(n_pixels, activation='relu', input_shape=(n_pixels,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(X_train.shape)
    plt.imshow(X_train[0])
    plt.show()

    n_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], n_pixels).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], n_pixels).astype('float32')

    X_train = X_train/255
    X_test = X_test/255

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    n_classes = y_test.shape[1]
    print(n_classes)

    model = classification(n_pixels, n_classes)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

    score = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: {} %\n Error Rate: {}".format(score[1], 1-score[1]))

    model.save('simple_classification_model.h5')

    #saved_model = load_model('simple_classification_model')


if __name__ == '__main__':
    main()
