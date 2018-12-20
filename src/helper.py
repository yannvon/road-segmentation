""" This file contains general helper functions """
import matplotlib.image as mpimg


def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

#source
#https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def cross_validate(epochs, folds = 4):
    """ Function used to crossvalidate our results """
    accuracy_sum = 0
    f1_sum = 0
    for i in range(folds):
        model.train(epochs=epochs, validation_split=0.2, seed=folds)
        scores = model.scores()
        accuracy_sum += scores[0]
        f1_sum += scores[1]
        print("Fold - "+str(i+1)+" : accuracy = "+str(scores[0])+", f1 = "+str(scores[1]))
        print("Cross-validation "+str(folds)+" folds : mean accuracy = "+str(accuracy_sum / folds)+", mean f1 score = "+str(f1_sum / folds))
