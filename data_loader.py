import pickle
import numpy as np
import cv2
import os

def laod_test_pk(org=False):
    if org:
        file = open('./data/test_or.pk', 'rb')
    else:
        file = open('./data/test.pk', 'rb')
    (X_train, y_train) = pickle.load(file)
    file.close()

    return X_train, y_train


def load_pk(org=False):
    file = open('./data/train.pk', 'rb')
    (X_train, y_train) = pickle.load(file)   # X_train sono le immagini , Y_train sono i valori della classe
    #print((X_train))
    #exit(0)
    #if org:
     #   file = open('./data/validation_or.pk', 'rb')
    #else:
     #   file = open('./data/validation.pk', 'rb')
    #(X_val, y_val) = pickle.load(file)
    file.close()

    return X_train, y_train


def load_images( name ) :
    print(name)

    lista_immagini = []
    gt = []
    dir_uomini = name+"male/"
    dir_donne = name+"female/"
    print("Man")
    for filename in os.listdir(dir_uomini):
        print(dir_uomini+filename)
        im = cv2.imread(dir_uomini+filename)
        print(im)
        lista_immagini.append(im)
        gt.append(0) #1 donna 0 uomo
    print("female")
    for filename in os.listdir(dir_donne):
        print(filename)
        im = cv2.imread(dir_donne+filename)
        lista_immagini.append(im)
        gt.append(1) #1 donna 0 uomo


    with open('train.pk', 'wb') as handle:
        pickle.dump((lista_immagini,gt),handle)



def save_image( npdata, outfilename ) :
    #img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    #img.save( outfilename )
    pass


#load_images(os.getcwd()+("/training_set_face/"))

#load_pk()


#file = open('train.pk', 'rb')
#(X_train,y_train) = pickle.load(file)
#print((X_train),(y_train))
#file.close()
