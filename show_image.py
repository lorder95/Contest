from data_loader import load_pk
import matplotlib.pyplot as plt


if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_pk()

    for i in range(X_train.shape[0]):
        img = X_train[i].reshape((28, 28))
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.show()
