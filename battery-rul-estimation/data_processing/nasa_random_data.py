import numpy as np
import pandas as pd
import logging
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import scipy.io

DATA_PATH = 'data/nasa-randomized/'
NOMINAL_CAPACITY = 2.2


class NasaRandomizedData():
    def __init__(self, base_path="./"):
        self.path = base_path + DATA_PATH
        self.logger = logging.getLogger()

    def get_discharge_whole_cycle_future(self, train_names, test_names):
        self.logger.info("Loading train data...")
        (train_x, train_y, battery_n_cycle_train,
            time_train, current_train) = self._get_data(train_names)
        self.logger.info("Loading test data...")
        (test_x, test_y, battery_n_cycle_test,
            time_test, current_test) = self._get_data(test_names)

        self.logger.info('''Train x: %s, train y soh: %s | Test x: %s, test y soh: %s | 
                            battery n cycle train: %s, battery n cycle test: %s, 
                            time train: %s, time test: %s |
                            raw current train: %s, raw current test: %s |
                            ''' %
                         (train_x.shape, train_y.shape, test_x.shape, test_y.shape,
                          battery_n_cycle_train.shape, battery_n_cycle_test.shape, 
                          time_train.shape, time_test.shape,
                          current_train.shape, current_test.shape))

        return (train_x, train_y, test_x, test_y,
                battery_n_cycle_train, battery_n_cycle_test,
                time_train, time_test,
                current_train, current_test)
        
    def _get_data(self, names):
        cycle_x = []
        cycle_y = []
        first_y = True
        y_between_count = 0
        battery_n_cycle = []
        time = []
        current = []
        n_cycles = 0
        max_step = 0
        for name in names:
            self.logger.info("Processing file %s" % name)
            raw_data = scipy.io.loadmat(self.path + name)['data'][0][0][0][0]
            cycle = pd.DataFrame(raw_data)

            cycle_num = 0
            cycle['cycle'] = cycle_num
            current_type = cycle.loc[0, 'type']
            for index in range(1, len(cycle.index)):
                if ((current_type == "C" and cycle.loc[index, 'type'] == "D") or 
                    (current_type == "D" and cycle.loc[index, 'type'] == "C") or
                    (current_type == "R" and cycle.loc[index, 'type'] != "R")):
                    current_type = cycle.loc[index, 'type']
                    cycle_num += 1
                cycle.loc[index, 'cycle'] = cycle_num

            for x in set(cycle["cycle"]):
                if cycle.loc[cycle["cycle"] == x, "type"].iloc[0] != "D":
                    continue

                cycle_x.append(np.column_stack([
                    np.hstack(cycle.loc[cycle["cycle"] == x, "voltage"].to_numpy().flatten()).flatten(),
                    np.hstack(cycle.loc[cycle["cycle"] == x, "current"].to_numpy().flatten()).flatten(),
                    np.hstack(cycle.loc[cycle["cycle"] == x, "temperature"].to_numpy().flatten()).flatten()]))

                n_cycles += 1
                step_time = np.hstack(cycle.loc[cycle["cycle"] == x, "time"].to_numpy().flatten()).flatten()
                time.append(step_time / 3600)
                current.append(np.hstack(cycle.loc[cycle["cycle"] == x, "current"].to_numpy().flatten()).flatten())
                max_step = max([max_step, cycle_x[-1].shape[0]])

                if (cycle.loc[cycle["cycle"] == x, "comment"].iloc[0] == "reference discharge" and
                     (x < 2 or cycle.loc[cycle["cycle"] == x-2, "comment"].iloc[0] != "reference discharge")):
                    current_y = np.trapz(current[-1], np.hstack(cycle.loc[cycle["cycle"] == x, "time"].to_numpy().flatten()).flatten())/3600
                    if y_between_count > 0:
                        step_y = (cycle_y[-1] - current_y)/y_between_count
                        while y_between_count > 0:
                            cycle_y.append(cycle_y[-1]-step_y)
                            y_between_count -=1
                    cycle_y.append(current_y)
                elif first_y is True:
                    cycle_y.append(NOMINAL_CAPACITY)
                else:
                    y_between_count += 1
                first_y = False

            while y_between_count > 0:
                cycle_y.append(cycle_y[-1])
                y_between_count -=1
            first_y = True
            battery_n_cycle.append(n_cycles)

        cycle_x = self._to_padded_numpy(cycle_x, [len(cycle_x), max_step, len(cycle_x[0][0])])
        cycle_y = np.array(cycle_y)
        battery_n_cycle = np.array(battery_n_cycle)
        time = self._to_padded_numpy(time, [len(time), max_step])
        current = self._to_padded_numpy(current, [len(current), max_step])

        return cycle_x, cycle_y, battery_n_cycle, time, current

    def _to_padded_numpy(self, l, shape):
        padded_array = np.zeros(shape)
        for i,j in enumerate(l):
            padded_array[i][0:len(j)] = j
        return padded_array


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    train_names = [
        'Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW1',
        #'Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW2',
        #'Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW7',

        #'Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW3',
        #'Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW4',
        #'Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW5',

        #'Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW9',
        #'Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW10',
        #'Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW11',

        #'RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW13',
        #'RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW14',
        #'RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW15',

        #'RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW17',
        #'RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW18',
        #'RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW19',

        #'RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW21',
        #'RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW22',
        #'RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW23',

        #'RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW25',
        #'RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW26',
        #'RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW27',

        ]
    test_names = [
        'Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW8',
        #'Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW6',
        #'Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW12',
        #'RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW16',
        #'RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW20',
        #'RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW24',
        #'RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW28',
        ]


    data = NasaRandomizedData()
    (train_x, train_y, test_x, test_y,
     battery_name_cycle_train, battery_name_cycle_test,
     time_train, time_test,
     current_train, current_test) = data.get_discharge_whole_cycle_future(train_names, test_names)





    VISUALIZATION_START = 0
    VISUALIZATION_END = 100000
    display_x = train_x.reshape(train_x.shape[0]*train_x.shape[1], train_x.shape[2])

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=display_x[VISUALIZATION_START:VISUALIZATION_END,0],
                        mode='lines', name='Voltage'))
    fig.update_layout(title='Voltage',
                    xaxis_title='Step',
                    yaxis_title='Voltage',
                    width=1400,
                    height=600)
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=display_x[VISUALIZATION_START:VISUALIZATION_END,1],
                        mode='lines', name='Current'))
    fig.update_layout(title='Current',
                    xaxis_title='Step',
                    yaxis_title='Current',
                    width=1400,
                    height=600)
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=display_x[VISUALIZATION_START:VISUALIZATION_END,2],
                        mode='lines', name='Temperature'))
    fig.update_layout(title='Temperature',
                    xaxis_title='Step',
                    yaxis_title='Temperature',
                    width=1400,
                    height=600)
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=time_train.flatten()[VISUALIZATION_START:VISUALIZATION_END],
                        mode='lines', name='Time'))
    fig.update_layout(title='Time',
                    xaxis_title='Step',
                    yaxis_title='Time',
                    width=1400,
                    height=600)
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_y.flatten()[VISUALIZATION_START:VISUALIZATION_END],
                        mode='lines', name='Capacity'))
    fig.update_layout(title='Capacity',
                    xaxis_title='Step',
                    yaxis_title='Capacity',
                    width=1400,
                    height=600)
    fig.show()