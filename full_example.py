import cv2
from keras.models import Sequential
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import SGD
from keras.optimizers import Adam

from keras.utils import to_categorical
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop
# User defined imports
from data_loader import load_pk
from MultiLayerPerceptron import PerceptronLayer
from score import score


def extract_hog_features(X):
    # Parameters
    winSize = (128,128)
    blockSize = (64, 64)
    blockStride = (32, 32)
    cellSize = (32,32)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64

    # Features extractor
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    output_dim = int(((winSize[0]-blockSize[0])/blockStride[0] +
                      1)**2 * nbins * (blockSize[0]/cellSize[0])**2)

    Xf = np.zeros((X.shape[0], output_dim))


    for i in range(X.shape[0]):

        img = cv2.resize(X[i],(int(128),int(128)))
        #cv2.imshow("image",img)
        #print(img.shape)


        #img = X[i].reshape((128, 128))
        Xf[i, :] = hog.compute(img)[:,0]

    # parameters dict
    params = {
        "winSize": winSize,
        "blockSize": blockSize,
        "blockStride": blockStride,
        "cellSize": cellSize,
        "nbins": nbins,
        "derivAperture": derivAperture,
        "winSigma": winSigma,
        "histogramNormType": histogramNormType,
        "L2HysThreshold": L2HysThreshold,
        "gammaCorrection": gammaCorrection,
        "nlevels": nlevels
    }

    return Xf, (output_dim,), params


if __name__ == "__main__":
    # Dataset loading
    data,labels= load_pk()

    X, Xv, y_labels, yv_labels = train_test_split(np.asarray(data), np.asarray(labels), test_size=0.3, shuffle= True)


    #print(y_labels[0],yv_labels[0])
    #newimg = cv2.resize(X[0],(int(128),int(128)))
    #cv2.imshow("image",newimg)
    #cv2.waitKey(0)
    #newimg = cv2.resize(Xv[0],(int(128),int(128)))
    #cv2.imshow("image",newimg)
    #cv2.waitKey(0)
    #exit(0)

    n_classes = 2

    # Features extraction
    X, input_shape, params_hog = extract_hog_features(X)
    Xv, _, _ = extract_hog_features(Xv)
    print("HOG Descriptor Size: ", input_shape)

    # Class 3 -> probability [0., 0., 0., 1.]
    y = to_categorical(y_labels)
    yv = to_categorical(yv_labels)

    # MLP parameters
    # The last value is the number of classes
    layer_sizes = [128,128, n_classes]
    # Available functions = step, tanh, sigmoid, softmax
    act_funcs = ['sigmoid', 'sigmoid', 'sigmoid']
    use_bias = [True, True, True]
    batch_size = 1  # Siccome il training è con Stochastic gradient lo farò solo sul batch size(ovvero non su tt il training set ) più grande è meglio approssima la f di errore ma farlo troppo grande potrebbe causare overfitting
    lr = 0.01       # Scala il valore del gradiente (evita l'overfitting)
    momentum = .9   # Evita i minimi locali , non uso il gradiente calcolato sul batch corrente , ma troppo grande rende più lento il trovare il minimo
    decay = .0001     # percentuale con cui diminuisco il Lr ad ogni epoca (Sempre per i minimi locali )
    epochs = 100
    patience = 15    # Stoppa l'addestramento se c'è overfitting

    # MLP model
    mlp = Sequential()
    # Input_shape only needed for the first layer
    mlp.add(PerceptronLayer(layer_sizes[0],
                            act=act_funcs[0], use_bias=use_bias[0], input_shape=input_shape))
    mlp.add(PerceptronLayer(layer_sizes[1],
                            act=act_funcs[1], use_bias=use_bias[1]))
    mlp.add(PerceptronLayer(layer_sizes[2],
                            act=act_funcs[2], use_bias=use_bias[2]))

    # MLP optimizer
    sgd_opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay, amsgrad=False)
    mlp.compile(optimizer=sgd_opt,
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # Training Callbacks
    #es = EarlyStopping(monitor='val_loss', min_delta=0,
     #                  patience=patience, restore_best_weights=True)
    #callback_list = [es]

    # MLP Training
    history = mlp.fit(X, y, batch_size, epochs, verbose=1,
                       validation_data=(Xv, yv))
     #callbacks=callback_list, DA INSERIRE COME PARAEMTRO IN HISTORY SE SI VUOLE FERMARE
    # Validation prediction
    # Probabilities
    yv_pred = mlp.predict(Xv)
    # Labels
    yv_lab_pred = np.argmax(yv_pred, axis=1)
    print("Validation Score w/o rejection: ", score(yv_labels, yv_lab_pred))

    # Plot training
    # Summarize history for accuracy
    t = [k+1 for k in range(len(history.history['acc']))]
    plt.plot(t, history.history['acc'])
    plt.plot(t, history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('./train_acc.png')
    plt.gcf().clear()

    # Summarize history for loss
    plt.plot(t, history.history['loss'])
    plt.plot(t, history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('./train_loss.png')
    plt.gcf().clear()

    # Rejection
    y_corr = np.max(yv_pred[np.where(yv_lab_pred == yv_labels)], axis=1)
    y_err = np.max(yv_pred[np.where(yv_lab_pred != yv_labels)], axis=1)

    print("Corrected classified samples: ", y_corr.shape[0])
    print("Wrong classified samples: ", y_err.shape[0])

    n_bins = 100
    bins = np.arange(n_bins+1)*1/n_bins

    # Correct confidence histogram
    n_corr, bins, patches = plt.hist(
        y_corr, bins=bins, facecolor='blue', alpha=0.5)

    plt.xlabel('Confidence')
    plt.ylabel('# of corrected classified samples')
    plt.title('Histogram')
    plt.xlim([0, 1])
    plt.grid(True)
    plt.savefig('./corr_hist.png')
    plt.gcf().clear()

    # Error confidence histogram
    n_err, bins, patches = plt.hist(
        y_err, bins=bins, facecolor='blue', alpha=0.5)

    plt.xlabel('Confidence')
    plt.ylabel('# of wrong classified samples')
    plt.title('Histogram')
    plt.xlim([0, 1])
    plt.grid(True)
    plt.savefig('./err_hist.png')
    plt.gcf().clear()

    # Threshold rejection
    scores = []
    for i in range(bins.shape[0]):
        yv_lab_pred_test = -1*np.ones_like(yv_lab_pred)
        idx = np.where(np.max(yv_pred, axis=1) >= bins[i])
        yv_lab_pred_test[idx] = yv_lab_pred[idx]

        scores.append(score(yv_labels, yv_lab_pred_test))

    scores = np.array(scores)
    th = bins[np.argmax(scores)]
    val_score = np.max(scores)

    print("Validation Score with rejection: ", val_score)
    print("Rejection th: ", th)

    # Plot of rejection th

    plt.hist(y_corr, bins=bins,
             density=True, stacked=True, facecolor='blue', alpha=0.3, label='Corr')
    plt.hist(y_err, bins=bins,
             density=True, stacked=True, facecolor='red', alpha=0.3, label='Err')
    plt.axvline(th, label='Rejection th', linestyle='--')

    plt.legend(loc='upper right')
    plt.xlabel('Confidence')
    plt.ylabel('% Normed corrected/wrong classified samples')
    plt.title('Histogram')
    plt.xlim([0, 1])
    plt.grid(True)
    plt.savefig('./th_hist.png')
    plt.gcf().clear()

    # Save model and parameters
    mlp.save('./model.h5')
    f = open('./params.pk', 'wb')
    pickle.dump({
        'th': th,
        'hog_params': params_hog
    }, f)
    f.close()
