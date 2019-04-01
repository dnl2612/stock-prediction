import math
import numpy as np
import pandas as pd

class DataProcessor():
    '''
    Loads and transforms data for the LSTM model
    '''

    def __init__(self, filename, split, columns):
        dataframe = pd.read_csv(filename)
        i_split = int(len(dataframe) * split)
        self.data_train = dataframe.get(columns).values[:i_split]
        self.data_test = dataframe.get(columns).values[i_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len, normalize):
        '''
        Creates test data windows
        '''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalize_windows(data_windows, single_window=False) if normalize else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x,y

    def get_train_data(self, seq_len, normalize):
        '''
        Creates train data windows
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalize)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_data(self, seq_len, data_size, normalize):
        '''
        Yields a generator of training data from filename
        '''
        i = 0
        while i < (self.len_train - seq_len):
            x_data = []
            y_data = []
            for b in range(data_size):
                if i >= (self.len_train - seq_len):
                    # If data isn't divided evenly, this is the stop condition for the last smaller part
                    yield np.array(x_data), np.array(y_data)
                    i = 0
                x, y = self._next_window(i, seq_len, normalize)
                x_data.append(x)
                y_data.append(y)
                i += 1
            yield np.array(x_data), np.array(y_data)

    def _next_window(self, i, seq_len, normalize):
        '''
        Generates the next data window from the given index location i
        '''
        window = self.data_train[i:i+seq_len]
        window = self.normalize_windows(window, single_window=True)[0] if normalize else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    def normalize_windows(self, window_data, single_window=False):
        '''
        Normalizes window with a base value of 0
        '''
        normalized_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalized_window = []
            for col_i in range(window.shape[1]):
                normalized_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalized_window.append(normalized_col)
            # Transpose array back into original multidimensional format
            normalized_window = np.array(normalized_window).T 
            normalized_data.append(normalized_window)
        return np.array(normalized_data)
