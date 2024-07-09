import numpy as np


class KNearestNeighbor():
    def __init__(self, trainingdata, traninglabels, n_neighbors):
        """
        :param trainingdata: Training data
        :param traninglabels: Training labels
        :param n_neighbors: number of neighbors
        """
        self.Data_Train = trainingdata
        self.Label_Train = traninglabels
        self.n_neighbors = n_neighbors

    def euclidian_distance(self, a, b):
        """
        Calculate the Euclidian distance between a and b.
        return: Euclidian distance between a and b.
        """
        euc_distance = np.sqrt(np.sum((a - b) ** 2, axis=1))  # use the horizontal axis to calculate

        return euc_distance

    def kneighbors(self, data):
        """
        Finds the k nearest neighbors of dataTest.
        :param data: Data point for which we want to find the k nearest neighbors.
        :return: k nearest neighbors of dataTest
        """
        neigh_ind = []

        for i in data:
            dist = self.euclidian_distance(i, self.Data_Train)
            en = enumerate(dist)
            so = sorted(en, key=lambda x: x[1])[:self.n_neighbors]

            tmp = [entry[0] for entry in so]
            neigh_ind.append(tmp)

        return np.array(neigh_ind)

    def predict(self, predict_data):
        """
        Predicts the label of TestData.
        :param predict_data: Data point-/s for which we want to predict the label.
        :return: predicted label-/s
        """
        neighbors = self.kneighbors(predict_data)
        preds = []
        for i in neighbors:
            prediction = np.argmax(np.bincount(self.Label_Train[i]))
            preds.append(prediction)

        return preds

    def accuracy(self, predicts, actual):
        """
        Calculates the accuracy of the predictions.
        :param predicts: Predicted data point-/s
        :param actual: Actual data point-/s
        :return: Accuracy of the predictions in percentage
        """
        diff = 0
        for idx, v in enumerate(predicts):
            if v != actual[idx]:
                diff += 1

        return (1 - diff / len(actual)) * 100
