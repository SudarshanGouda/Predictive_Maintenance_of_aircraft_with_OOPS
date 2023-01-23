import pandas as pd
import pickle
import warnings

warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler


class airclaft_predictive_model():
    '''
    A class to represent a predictive maintainance of aircraft.

    ...

    Attributes
    ----------
    model file : file
                predictive model saved after the experiment
            In this case it is the 'Neural Network Regression' model

    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
    pass'''

    def __init__(self, model_file):

        """
        Constructs all the necessary attributes for the person object.

        Parameters
        ----------
            model file : file
                predictive model saved after the experiment
            In this case it is the 'Neural Network Regression' model
        """
        # read the 'model' files which were saved
        with open('model', 'rb') as model_file:
            self.regression = pickle.load(model_file)

    def load_clean_data(self, data_file):
        # take a data file (*.txt) and preprocess it
        """
            Import the text file, and it processes and clean and standardize the file required for prediction
        Parameters
        ----------
        data_file : in .txt format

        Returns
        -------
        cleaned and processed file required for prediction
        """
        # import the data
        df = pd.read_csv(data_file, sep=' ', header=None)

        # Column Names
        col_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8',
                     's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22',
                     's23']
        df.columns = col_names

        # store the data in a new variable for later use
        self.df_with_predictions = df.copy()

        # Droping the colums
        df.drop(['setting1', 'setting2', 's6', 'setting3', 's1', 's5', 's10', 's16', 's18', 's19', 's22', 's23'],
                axis=1, inplace=True)

        df = df.drop(['id'], axis=1)

        self.preprocessed_data = df.copy()

        sc = StandardScaler()

        self.data = sc.fit_transform(df)

    # a function which outputs the probability of a data point to be 1
    def predicted_vallue(self):
        """
            Processed data will be predicted.
        ----------

        Returns
        -------
        Predicted values
        """
        if (self.data is not None):
            pred = self.regression.predict(self.data)[:, 1]
            return pred

    # predict the outputs and
    # add columns with these values at the end of the new data

    def predicted_outputs(self):
        '''
        Processed da will be predicted and concat with table
        :return:
        '''

        if (self.data is not None):
            self.prediction = self.regression.predict(self.data)
            self.preprocessed_data['Prediction'] = self.prediction
            return self.preprocessed_data