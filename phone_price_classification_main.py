import tkinter as tk
from tkinter import ttk
from tksheet import Sheet
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import knn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sb
import collections


class Window:
    def __init__(self):
        """
        Constructor for the main window
        """
        self.selectedId = None  # contains the selectedId id from the dropbox
        self.selectedData = []  # contains the information behind the selectedId id

        self.data_preprocessing()

        # Initialize KNN with training data
        self.knn = knn.KNearestNeighbor(self.traincsv, self.price_range, 9)

        # Set up the main window
        self.root = tk.Tk()
        self.root.title("Phone Price Calculation")

        # Set window size
        self.root.geometry("500x500")
        self.show_main_window()
        self.root.mainloop()

    def data_preprocessing(self):
        """
        Import and Preprocessing the Data from the csv files
        :return: None
        """
        # Load test.csv into a Pandas DataFrame and convert it to a Numpy array
        test_file = pd.read_csv('test.csv')
        self.testcsv = np.array(test_file)
        self.test_data = np.delete(self.testcsv, 0, axis=1)

        # Load train.csv and split the data into Data and labels
        train_file = pd.read_csv('train.csv')
        price_range = train_file["price_range"]
        self.price_range = np.array(price_range)
        self.traincsv = np.array(train_file.iloc[:, :-1])

    def show_selected(self, event=None):
        """
        Event handler for selecting an item from the combobox.
        """
        self.selectedId = self.selectCombobox.get()
        self.show_sheet()

    def show_sheet(self, label=None):
        """
        Displays the selectedId sheet in the window, creating a table with the selectedId data.
        :param label: Value shown at price tagging.
        :return: False if selectedId is not an integer.
        """
        # Appending on selected there are different ways to show the data in the table
        if self.selectedId == "all":
            self.selectedData = self.testcsv
            if label is None or all(item == "" for item in label):
                self.sheet = Sheet(self.bottomFrame,
                                   data=[[f"{x}" for x in self.selectedData[i]] +
                                         [""] for i in range(len(self.selectedData))],
                                   width=480, height=390)
            else:
                self.sheet = Sheet(self.bottomFrame, data=[[f"{x}" for x in self.selectedData[i]] + [label[i]] for i in
                                                           range(len(self.selectedData))], width=480, height=390)
            self.sheet.row_index([int(self.selectedData[i][0]) for i in range(len(self.selectedData))])
        else:
            try:
                self.selectedId = int(self.selectedId)
            except ValueError:
                return False
            self.selectedData = self.testcsv[self.selectedId - 1]
            self.sheet = Sheet(self.bottomFrame,
                               data=[[f"{x}" for x in self.selectedData] + [label]], width=480, height=100)
            self.sheet.row_index([int(self.selectedData[0])])

        # format the table and set headers
        self.sheet.headers(
            ["id", "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g", "int_memory", "m_dep",
             "mobile_wt", "n_cores", "pc", "px_height", "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g",
             "touch_screen", "wifi", "price_range"])
        self.sheet.del_column(0)
        self.sheet.grid(row=1, column=0, sticky="nsew")

    def predict(self):
        """
        Predicts the price range for the selectedId data using KNN.
        """
        predict_data = self.selectedData

        # Reshape the data if necessary
        if self.selectedId != "all":
            predict_data = predict_data.reshape(1, -1)

        # delete the ids
        input_data = np.delete(predict_data, 0, axis=1)

        # predict labels for selected
        predicted_label = self.knn.predict(input_data)
        if len(predicted_label) == 1:
            predicted_label = predicted_label[0]

        # Update table
        self.update_sheet(predicted_label)

    def update_sheet(self, predicted_label):
        """
        Updates the sheet with the predicted labels
        :param predicted_label: list with the predicted labels
        """
        self.sheet.delete()
        self.sheet.destroy()
        self.show_sheet(predicted_label)

    def test(self, best_nearest_neighbour=9):
        """
        Tests the accuracy of the KNN model using a train-test split.
        """
        #Split the data int train and test data
        data_train, data_test, label_train, label_test = train_test_split(self.traincsv, self.price_range,
                                                                          test_size=0.2, random_state=45)
        self.knn_test = knn.KNearestNeighbor(data_train, label_train, best_nearest_neighbour)
        pred = self.knn_test.predict(data_test)

        # detect errors and calculate accuracy
        error = 0
        for i in range(len(pred)):
            print(f"Predicted/Actual: {int(pred[i]), int(label_test[i])}")
            if pred[i] != label_test[i]:
                error += 1
                print("error!!")
        acc = self.knn.accuracy(pred, label_test)

        # show the accuracy in a label
        accuracy_label = tk.Label(self.topFrame, text=f"Accuracy: {acc}%")
        accuracy_label.grid(row=0, column=1, sticky="nsew")
        print(f"Test wrong predicted: {error} of 400")
        print(f"Accuracy with {best_nearest_neighbour} nearest neighbours: {acc}%")

    def show_main_window(self):
        """
        Shows the widgets in the window.
        """
        # create segments in window
        self.topFrame = tk.Frame(self.root)
        self.topFrame.pack()
        self.middleFrame = tk.Frame(self.root)
        self.middleFrame.pack()
        self.bottomFrame = tk.Frame(self.root)
        self.bottomFrame.pack()

        # button accuracy test
        test_button = ttk.Button(self.topFrame, text="Accuracy Test", command=self.test)
        test_button.grid(column=0, row=0)

        # Label
        text = tk.Label(self.topFrame, text="From test.csv, open:", padx=5, pady=10)
        text.grid(column=0, row=1)

        # Drop down menu to select ID
        self.selectCombobox = ttk.Combobox(self.topFrame,
                                           values=["all"] + [str(x) for x in range(1, len(self.testcsv) + 1)])
        self.selectCombobox.bind("<<ComboboxSelected>>", self.show_selected)
        self.selectCombobox.grid(column=1, row=1)

        # button Price Detecting
        searchPricingButton = ttk.Button(self.middleFrame, text="Detect Pricing", padding=10, command=self.predict)
        searchPricingButton.grid(column=1, row=0)

        """
            The following Code is only for testing purposes.
            This is only to show what was done in the documentation 
        """
        '''
        self.docFrame = tk.Frame(self.root)
        self.docFrame.pack()

        Acc_button = ttk.Button(self.docFrame, text="Documentation Graphs", command=self.graphs_and_dependency)
        Acc_button.grid(column=0, row=0)
        Acc_label = tk.Label(self.docFrame, text="The results are not shown in GUI", padx=5, pady=10)
        Acc_label.grid(column=1, row=0)

    def detect_best_nn(self):
        """
        Detects the best nuber of nearest neighbours to get the highest accuracy
        """
        x = []
        for i in range(200):
            tmp = self.highest_acc_of_nn(random_seed=i)
            try:
                if type(tmp) == list:
                    for m in tmp:
                        x.append(m)
                elif type(tmp) == int:
                    x.append(tmp)
            except:
                print("There was an error while trying to find the best nn")
        print(f"Values of KNN model: {x}")

        counter = collections.Counter(x)
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(list(counter.keys()), list(counter.values()), label="nearest neighbour", marker=".", markersize=20,
                linestyle="None")
        ax.set_xticks(list(counter.keys()))
        plt.title("Best Accuracy based on nearest neighbour")
        plt.ylabel("Amount")
        plt.xlabel("n nearest neighbours")
        plt.show()
        
        # Get Heatmap for 9NN and 400 Test data 
        data_train, data_test, label_train, label_test = train_test_split(self.traincsv, self.price_range,
                                                                          test_size=0.2, random_state=45)
        self.knn_test = knn.KNearestNeighbor(data_train, label_train, 9)
        pred = self.knn_test.predict(data_test)
        matrix = confusion_matrix(label_test, pred)
        print(f"\nThe Matrix looks like:\n {matrix}")
        sb.heatmap(matrix, annot=True)
        plt.show()

    def highest_acc_of_nn(self, n_nearest_neighbors=40, random_seed=45):
        """
        Tests the accuracy of the KNN model using a train-test split and returns the best accuracy. Creating a histogram
        :param n_nearest_neighbors: Number of nearest neighbors to test the accuracy of.
        :return: nearest neighbour with best test accuracy
        """
        data_train, data_test, label_train, label_test = train_test_split(self.traincsv, self.price_range,
                                                                          test_size=0.2, random_state=random_seed)
        accuracy_test = []
        for nn in range(1, n_nearest_neighbors):
            self.knn_test = knn.KNearestNeighbor(data_train, label_train, nn)
            pred = self.knn_test.predict(data_test)
            acc = self.knn.accuracy(pred, label_test)
            accuracy_test.append(acc)
        max_value = accuracy_test[np.argmax(accuracy_test)]
        max_idx = []
        for id, d in enumerate(accuracy_test):
            if d == max_value:
                max_idx.append(id + 1)
        return max_idx


    def graphs_and_dependency(self):
        """
        Creates the graphs and dependency graphs that are used in the documentation
        """
        print(f"The complete Task can take a lot of time...")
        train_file = pd.read_csv('train.csv')
        labels = self.price_range
        data = self.traincsv

        # Attempt 1: devide data by biggest
        print("\nPlot of attempt 1 is in progress")
        max_value = []
        for i in range(20):
            max_value.append(np.max(data[:, i]))
        print(f"Max values in each features: {[int(num) for num in max_value]}")
        calc = []
        for j in range(len(data)):
            calc.append(data[j] / max_value)
        self.get_accuracy_of_one_randomstate(calc, labels, "Accuracy")

        # Attempt 2: standard deviation
        print("\nPlot of attempt 2 is in progress")
        std_data = ((data - np.mean(data, axis=0)) / np.std(data, axis=0))
        self.get_accuracy_of_one_randomstate(std_data, labels, "Accuracy")

        # Attempt 3: Do nothing
        print("\nPlot of attempt 3 is in progress")
        self.get_accuracy_of_one_randomstate(data, labels, "Accuracy")

        print("\nHeatmap of data correlation is in progress")
        plt.figure(figsize=(15, 8))
        corr = train_file.corr()
        sb.heatmap(corr, annot=True, cmap="RdPu")
        plt.show()

        # Detect best number of nearest neighbours
        print("\nPlot of best nearest neighbour is in progress, this need a lot of time (about 20-30 minutes)")
        self.detect_best_nn()

    def get_accuracy_of_one_randomstate(self, dataset, labels, title):
        """
        Calculates the accuracy of the KNN model using a train-test split with only one random state.
        :param dataset: Complete dataset without labels
        :param labels: labels of the dataset
        :param title: Title of the graph
        """
        acc = []
        data_train, data_test, label_train, label_test = train_test_split(dataset, labels,
                                                                          test_size=0.2, random_state=45)
        for nn in range(1, 200):
            self.knn_test = knn.KNearestNeighbor(data_train, label_train, nn)
            pred = self.knn_test.predict(data_test)
            error = 0
            for i in range(len(pred)):
                if pred[i] != label_test[i]:
                    error += 1
            acc.append(self.knn.accuracy(pred, label_test))
        min_value = acc[np.argmin(acc)]
        max_value = acc[np.argmax(acc)]
        acc.insert(0, 0)
        plt.plot(acc)
        plt.xlim(1, 200)
        plt.ylim(min_value - 2, max_value + 2)
        plt.title(title)
        plt.xlabel("n nearest neighbours")
        plt.ylabel("accuracy in %")
        max_idx = np.argmax(acc)

        # Get accuracy of best number of nearest neighbours
        self.knn_test = knn.KNearestNeighbor(data_train, label_train, max_idx)
        acc = self.knn.accuracy(self.knn_test.predict(data_train), label_train)
        print(f"max accuracy at {max_idx} nearest neighbours")
        print(f"Training Accuracy: {acc}%")
        acc = self.knn.accuracy(self.knn_test.predict(data_test), label_test)
        print(f"Test Accuracy: {acc}%")

        plt.show()
        '''


if __name__ == "__main__":
    Window()
