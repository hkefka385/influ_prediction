import pandas as pd
import numpy as np

#area: ALL, 10Area, 9Area, states
#data_t: 1, 2
def data_load(data_type, data_t):
    if data_t == 1:
        file_loc = 'DATA/' + data_type + '/ILINet.csv'
        with open(file_loc, mode='rb') as f:
            data = pd.read_csv(f, header=1)
        weighted_ili = list(data.iloc[:, 4])
        date = [str(data['YEAR'][i]) + '_' + str(data['WEEK'][i]) for i in range(len(data))]
        last_zero = 0
        for i, value in enumerate(weighted_ili):
            if value == 0.0:
                last_zero = i
        weighted_ili = np.array(weighted_ili[(last_zero + 1):], dtype=float)
        date = np.array(date[(last_zero + 1):])
        return weighted_ili, date
