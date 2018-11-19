#!/usr/bin/env python



##############

#### Your name: Alexis Vincent

##############



import numpy as np

import re

from skimage.color import convert_colorspace
from sklearn.model_selection import GridSearchCV

from sklearn import svm, metrics

from skimage import io, feature, filters, exposure, color

from skimage.feature import hog



import matplotlib.pyplot as plt





class ImageClassifier:

    def __init__(self):

        self.classifer = None



    def imread_convert(self, f):

        return io.imread(f).astype(np.uint8)



    def load_data_from_folder(self, dir):

        # read all images into an image collection

        ic = io.ImageCollection(dir + "*.jpg", load_func=self.imread_convert)



        # create one large array of image data

        data = io.concatenate_images(ic)

        # extract labels from image names

        labels = np.array(ic.files)

        for i, f in enumerate(labels):

            m = re.search("_", f)

            labels[i] = f[len(dir):m.start()]



        return (data, labels)



    def extract_image_features(self, data):

        # Please do not modify the header above

        # extract feature vector from image data

        fd = None

        for pic in data:

            #grey_picture = color.rgb2gray(pic)

            #gaussian_picture = filters.gaussian(pic, 1)

            rescaled_picture = exposure.rescale_intensity(pic)



            feature_data = hog(rescaled_picture,

                               orientations=11,

                               #pixels_per_cell=(32, 32),
                               pixels_per_cell=(20, 20),
                               cells_per_block=(6, 6),

                               # transform_sqrt=True,

                               feature_vector=True,

                               block_norm='L2-Hys')

            # self.print_hog_pics(color.rgb2gray(gaussian_picture))

            if fd is None:

                fd = feature_data.reshape(1, feature_data.shape[0])

            else:

                fd = np.concatenate([fd, feature_data.reshape(1, feature_data.shape[0])])

        # Please do not modify the return type below

        return fd



    def train_classifier(self, train_data, train_labels):

        # Please do not modify the header above

        # train model and save the trained model to self.classifier

        clf = svm.SVC(C=1, gamma=0.001, kernel='linear')

        self.classifer = clf.fit(train_data, train_labels)



    def predict_labels(self, data):

        # Please do not modify the header

        # predict labels of test data using trained model in self.classifier

        # the code below expects output to be stored in predicted_labels

        predicted_labels = self.classifer.predict(data)

        # Please do not modify the return type below

        return predicted_labels



    def print_hog_pics(self, image):
        #orientations=8, pixels_per_cell=(16, 16) cells_per_block=(1, 1), visualise=True
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),

                            cells_per_block=(1, 1), visualise=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex='all', sharey='all')



        ax1.axis('off')

        ax1.imshow(image)

        ax1.set_title('Input image')

        ax1.set_adjustable('box-forced')



        # Rescale histogram for better display

        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))



        ax2.axis('off')

        ax2.imshow(hog_image_rescaled)

        ax2.set_title('Histogram of Oriented Gradients')

        ax1.set_adjustable('box-forced')

        plt.show()





def main():

    img_clf = ImageClassifier()



    # load images

    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')

    (test_raw, test_labels) = img_clf.load_data_from_folder('./test/')



    # convert images into features

    train_data = img_clf.extract_image_features(train_raw)

    test_data = img_clf.extract_image_features(test_raw)



    # train model and test on training data

    img_clf.train_classifier(train_data, train_labels)



    predicted_labels = img_clf.predict_labels(train_data)

    print("\nTraining results")

    print("=============================")

    print("Confusion Matrix:\n", metrics.confusion_matrix(train_labels, predicted_labels))

    print("Accuracy: ", metrics.accuracy_score(train_labels, predicted_labels))

    print("F1 score: ", metrics.f1_score(train_labels, predicted_labels, average='micro'))
    print(predicted_labels)



    # test model

    predicted_labels = img_clf.predict_labels(test_data)

    print("\nTesting results")

    print("=============================")

    print("Confusion Matrix:\n", metrics.confusion_matrix(test_labels, predicted_labels))

    print("Accuracy: ", metrics.accuracy_score(test_labels, predicted_labels))

    print("F1 score: ", metrics.f1_score(test_labels, predicted_labels, average='micro'))
    print(predicted_labels)





if __name__ == "__main__":

    main()