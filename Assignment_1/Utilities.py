from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

class Utils: 

    # Load Data Set 1
    def load_dataset_1():
        data1 = datasets.load_iris()
        feature_names = ['fSepalLength', 'fSepalWidth', 'fPedalLength', 'fPedalWidth']
        target_names = ['tSetosa', 'tVersicolour', 'tVirginica']
        X, y = data1.data, data1.target
        return (X, y, feature_names, target_names)

    # Load Data Set 2
    def load_dataset_2():
        data2 = datasets.load_wine()
        feature_names = ['fAlcohol', 'fMalicAcid', 'fAsh', 'fAshAlcalinity', 'fMagnesium',
                         'fPhenols', 'fFlavanoids', 'NonflavPhenols', 'fProanthocyanins', 
                         'fColorIntensity', 'fHue', 'fOD280_OD315', 'fProline']
        target_names = ['tClass_O', 'tClass_1', 'tClass_2']
        X, y = data2.data, data2.target
        return (X, y, feature_names, target_names)
    
    # Save Data
    def save_data(filename, data, row_titles, col_titles):
        dataframe = pd.DataFrame(data, row_titles, col_titles)
        dataframe.to_pickle(filename + ".pkl")
        
    # Load Data
    def load_data(filename):
        data = pd.read_pickle(filename + ".pkl")
        return data
    
    # Plot Data Distribution
    def plot_data_distribution(targetNames, targetCount):
        plt.style.use('ggplot')

        x = targetNames
        data_distro = np.array(list(targetCount.values()))
        y = data_distro/np.sum(data_distro)

        x_pos = [i for i, _ in enumerate(x)]

        plt.bar(x_pos, y, color='green')
        plt.xlabel("Target Type")
        plt.ylabel("Data Distribution %")
        plt.title("Data Distribution")

        plt.xticks(x_pos, x)

        plt.show()
        # Find number of unique features and lables