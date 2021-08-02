import numpy as np
import keras
import os


class CustomDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_dir, batch_size=5, dim=(224,224), n_channels=3, should_shuffle = True):
        self.data_dir = data_dir
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.labels = {}
        self.files = []
        # Assuming class 'NORMAL' class is 0 and 'PNEMONIA' class is 1:
        for file in os.listdir(data_dir + "/NORMAL"):
            self.labels[file] = 0
            self.files.append(file)
        for file in os.listdir(data_dir + "/PNEUMONIA"):
            self.labels[file] = 1
            self.files.append(file)
        self.should_shuffle = should_shuffle
        self.on_epoch_end()

    # Gives us the number of batches per epoch. When a generator is passed to 'model.fit_generator(...)'
    # this function first calls '__len__()' to get the number number of batches per epoch.
    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))

    # Returns one batch of files. Is called automatically every time a new batch is needed.
    # By '__len__()' the caller knows the amount of batches to call, hence calls '__getitem__()' this amount of times.
    # The argument 'batch_index' is a counter for which number batch is requested.
    # This passed by the calling function which will keep track of it.
    def __getitem__(self, batch_index):
        files_in_batch = self.files[batch_index*self.batch_size:(batch_index+1)*self.batch_size]
        X, y = self.__data_generation(files_in_batch)

        return X, y

    # This function is run after each epoch. It shuffles the validation data in case we wish to do so,
    # even though it makes no sense. However, it can also serve as a blueprint in case we were to
    # make a generator with a shuffle option for training data.
    def on_epoch_end(self):
        if self.should_shuffle:
            np.random.shuffle(self.files)

    # Returns X : (batch_size, *dim, n_channels) and labels y.
    def __data_generation(self, files_in_batch):
        # Construct containers of X and y with the correct dimensions
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Fill the containers. Since the .txt-files are mono-channel we must first convert them to tri-channel.
        # We do this one-by-one as we add the files to the containers.
        for i, file in enumerate(files_in_batch):
            if self.labels[file] == 0:
                mono_channel_img = np.loadtxt(self.data_dir + "/NORMAL/" + file)
                X[i,] = np.expand_dims(np.stack((mono_channel_img,)*3, axis=-1), axis=0)
                y[i] = self.labels[file]
            else:
                mono_channel_img = np.loadtxt(self.data_dir + "/PNEUMONIA/" + file)
                X[i,] = np.expand_dims(np.stack((mono_channel_img,)*3, axis=-1), axis=0)
                y[i] = self.labels[file]
        return X, y


