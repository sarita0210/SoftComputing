import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.datasets import mnist
import cv2


def iseci_broj(broj):
    _, siva = cv2.threshold(broj, 128, 255, cv2.THRESH_BINARY)
    _, konture, _ = cv2.findContours(siva, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #povrsine kontura nadjenih na slici broja
    povrsine = []
    for kontura in konture:
        povrsine.append(cv2.contourArea(kontura))
    if len(povrsine) == 0:
        return broj
    najveca = 0
    #trazi najvecu
    for i,povrs in enumerate(povrsine):
        if povrs > povrsine[najveca]:
            najveca = i
    [x, y, w, h] = cv2.boundingRect(konture[najveca])
    isecena = broj[y:y + h + 1, x:x + w + 1]
    isecena = cv2.resize(isecena, (28,28), interpolation=cv2.INTER_AREA)
    return isecena
def ispraviSlova(img):
    velicina = 28
    moments = cv2.moments(img)
    if abs(moments['mu02']) < 1e-2:
        return img.copy()
    nakrivljenost = moments['mu11'] / moments['mu02']
    M = np.float32([[1, nakrivljenost, -0.5 * velicina * nakrivljenost], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (velicina, velicina), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img
#https://www.learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/
def napravi_model(oblik, broj_klasa=10):
    model = Sequential()
    model.add(Conv2D(28, (3, 3), padding='same', activation='relu', input_shape=oblik))
    model.add(Conv2D(28, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(56, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(56, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(56, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(56, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(broj_klasa, activation='softmax'))

    return model


if __name__ == '__main__':
    (tr_slike, tr_labele), (te_slike, te_labele) = mnist.load_data()
    #
    broj_klasa = 10

    for i in range(len(te_slike)):
        isecena = iseci_broj(te_slike[i])
        te_slike[i] = isecena
    for i in range(len(tr_slike)):
        isecena = iseci_broj(tr_slike[i])
        tr_slike[i] = isecena


    red, kolona = tr_slike.shape[1:]
    tr_podaci = tr_slike.reshape(tr_slike.shape[0], red, kolona, 1)
    te_podaci = te_slike.reshape(te_slike.shape[0], red, kolona, 1)
    oblik = (red, kolona, 1)

    tr_podaci = tr_podaci.astype('float32')
    te_podaci = te_podaci.astype('float32')
    # Scale the data to lie between 0 to 1
    tr_podaci /= 255
    te_podaci /= 255

    # iz int u kategoricki
    tr_lab_kat = to_categorical(tr_labele)
    te_lab_kat = to_categorical(te_labele)

    model = napravi_model(oblik, broj_klasa)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    istorija = model.fit(tr_podaci, tr_lab_kat, batch_size=256, epochs=30, verbose=1,
                             validation_data=(te_podaci, te_lab_kat))
    gubitak, tacnost = model.evaluate(te_podaci, te_lab_kat, verbose=0)
    model.save_weights('weights.h5')
    print(tacnost)




