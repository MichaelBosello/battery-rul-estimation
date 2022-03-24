import logging

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .unibo_powertools_data import CapacityCols, CycleCols


class ModelDataHandler():
    def __init__(self, dataset, x_indices, scaler_type=MinMaxScaler):
        self.logger = logging.getLogger()
        self.dataset = dataset
        self.x_indices = x_indices
        self.scaler_type = scaler_type

        self.train_charge_cyc, self.train_charge_cap, self.test_charge_cyc, self.test_charge_cap = self.dataset.get_charge_data()
        self.train_discharge_cyc, self.train_discharge_cap, self.test_discharge_cyc, self.test_discharge_cap = self.dataset.get_discharge_data()

        self.__assign_scalers()

    def __assign_scalers(self):
        self.charge_scalers = self.__create_scalers(self.train_charge_cyc)
        self.discharge_scalers = self.__create_scalers(
            self.train_discharge_cyc)

    def __create_scalers(self, cyc):
        scalers = []
        for index in self.x_indices:
            scalers.append(self.__create_scaler(cyc, index))
        return scalers

    def __create_scaler(self, cyc, col_index):
        data = np.concatenate(cyc)[:, col_index].reshape(-1, 1)
        scaler_x = self.scaler_type()
        scaler_x.fit_transform(data)
        return scaler_x

    def get_scalers(self):
        return self.charge_scalers, self.discharge_scalers

    def get_discharge_whole_cycle(self, output_capacity=False, multiple_output=False, soh=False):
        """x: [ [[voltage, current, temperature], ...], ...] \n
        SOH y (single step): [ [soh/last_charging_capacity, remaining_time_to_cell_end], ... ] \n
        SOH y (multiple steps): [ [[soh/last_charging_capacity, remaining_time_to_cell_end], ...], ... ]\n
        SOC y: [[[soc/discharging_capacity, remaining_time_to_cycle_end], ...], ...]"""

        if(soh):
            y_indices = [
                CapacityCols.CORRESPONDING_CHARGING_CAPACITY if output_capacity else CapacityCols.SOH,
                CapacityCols.REMAINING_TIME_TO_CELL_END
            ]
            train_raw_x, train_y = self.__get_whole_cycle_soh_x_y(
                self.train_discharge_cyc, self.train_discharge_cap, self.x_indices, y_indices
            )
            test_raw_x, test_y = self.__get_whole_cycle_soh_x_y(
                self.test_discharge_cyc, self.test_discharge_cap, self.x_indices, y_indices
            )
        else:
            y_indices = [
                CycleCols.DISCHARGING_CAPACITY if output_capacity else CycleCols.SOC,
                CycleCols.REMAINING_TIME_TO_CYCLE_END
            ]
            train_raw_x, train_y = self.__get_whole_cycle_soc_x_y(
                self.train_discharge_cyc, self.x_indices, y_indices
            )
            test_raw_x, test_y = self.__get_whole_cycle_soc_x_y(
                self.test_discharge_cyc, self.x_indices, y_indices
            )

        train_scaled_x = self.__get_scaled_whole_cycle_x(
            train_raw_x, self.discharge_scalers)
        test_scaled_x = self.__get_scaled_whole_cycle_x(
            test_raw_x, self.discharge_scalers)

        train_x, test_x = self.__get_padded_whole_cycle(
            train_scaled_x, test_scaled_x)
        if(not soh):
            train_y, test_y = self.__get_padded_whole_cycle(train_y, test_y)

        if(multiple_output and soh):
            # (SOH only) duplicate the y values to multiple steps for each cycle
            train_y = np.repeat(train_y[:, None, :], train_x.shape[1], axis=1)
            test_y = np.repeat(test_y[:, None, :], test_x.shape[1], axis=1)

        self.logger.info("Train x: %s, train y: %s | Test x: %s, test y: %s" %
                         (train_x.shape, train_y.shape, test_x.shape, test_y.shape))

        return (train_x, train_y, test_x, test_y)

    def __get_whole_cycle_soh_x_y(self, cyc, cap, x_indices, y_indices):
        x = np.array(
            list(map(lambda data: data[:, x_indices].astype('float32'), cyc))
        )
        y = np.array(cap[:, y_indices], dtype='float32')
        return (x, y)

    def __get_whole_cycle_soc_x_y(self, cyc, x_indices, y_indices):
        x = np.array(
            list(map(lambda data: data[:, x_indices].astype('float32'), cyc))
        )
        y = np.array(
            list(map(lambda data: data[:, y_indices].astype('float32'), cyc))
        )
        return (x, y)

    def __get_scaled_whole_cycle_x(self, x, scalers):
        def map_func(data):
            result = []
            for i in range(len(scalers)):
                result.append(scalers[i].transform(data[:, [i]]).flatten())
            return np.array(result).T
        return np.array(list(map(map_func, x)))

    def __get_padded_whole_cycle(self, train, test, min_cycle_length=None):
        max_cycle_step_count = max(len(cycle)
                                   for cycle in np.append(train, test))
        required_step_count = max_cycle_step_count
        if min_cycle_length is not None:
            required_step_count = max(max_cycle_step_count, min_cycle_length)

        def padding_map_func(data):
            pad_width = ((0, required_step_count - len(data)), (0, 0))
            return np.pad(data, pad_width, 'constant', constant_values=0)

        train_padded = np.array(list(map(padding_map_func, train)))
        test_padded = np.array(list(map(padding_map_func, test)))

        return (train_padded, test_padded)

    def get_discharge_single_step(self, output_capacity=False, soh=False):
        """x: [[voltage, current, temperature], ...]\n
           SOH y: [[soh/last_charging_capacity, remaining_time_to_cell_end], ...]
           SOC y: [[soc/discharging_capacity, remaining_time_to_cycle_end], ...]"""

        if(soh):
            y_indices = [
                CapacityCols.CORRESPONDING_CHARGING_CAPACITY if output_capacity else CapacityCols.SOH,
                CapacityCols.REMAINING_TIME_TO_CELL_END
            ]
            train_x, train_y = self.__get_single_step_soh(
                self.train_discharge_cyc, self.train_discharge_cap, y_indices)
            test_x, test_y = self.__get_single_step_soh(
                self.test_discharge_cyc, self.test_discharge_cap, y_indices)
        else:
            y_indices = [
                CycleCols.DISCHARGING_CAPACITY if output_capacity else CycleCols.SOC,
                CycleCols.REMAINING_TIME_TO_CYCLE_END
            ]
            train_x, train_y = self.__get_single_step_soc(
                self.train_discharge_cyc, y_indices)
            test_x, test_y = self.__get_single_step_soc(
                self.test_discharge_cyc, y_indices)

        for i in range(len(self.discharge_scalers)):
            train_x[:, [i]] = self.discharge_scalers[i].transform(
                train_x[:, [i]])
            test_x[:, [i]] = self.discharge_scalers[i].transform(
                test_x[:, [i]])

        self.logger.info("Train x: %s, train y: %s | Test x: %s, test y: %s" %
                         (train_x.shape, train_y.shape, test_x.shape, test_y.shape))

        return (train_x, train_y, test_x, test_y)

    def __get_single_step_soc(self, cyc, y_indices):
        concatenated_cyc = np.concatenate(cyc)
        x = concatenated_cyc[:, self.x_indices].astype('float32')
        y = concatenated_cyc[:, y_indices].astype('float32')
        return (x, y)

    def __get_single_step_soh(self, cyc, cap, y_indices):
        x = list()
        y = list()
        for i in range(len(cyc)):
            current_cyc = cyc[i]
            current_cap = cap[i]
            x.extend(current_cyc[:, self.x_indices])
            y.extend(np.repeat([current_cap[y_indices]],
                     len(current_cyc), axis=0))
        x = np.array(x).astype('float32')
        y = np.array(y).astype('float32')
        return (x, y)

    def get_discharge_multiple_step(self, steps, output_capacity=False, multiple_output=False, soh=False):
        """x: [[[voltage, current, temperature], ...] ...]\n
           SOH y (single step): [[soh/last_charging_capacity, remaining_time_to_cell_end], ...]\n
           SOH y (multiple steps): [[[soh/last_charging_capacity, remaining_time_to_cell_end], ...], ...]\n
           SOC y (single step): [[soc/discharging_capacity, remaining_time_to_cycle_end], ...]\n
           SOC y (multiple steps): [[[soc/discharging_capacity, remaining_time_to_cycle_end], ...], ...]\n
           (Pad zeros to begining steps)"""

        if(soh):
            y_indices = [
                CapacityCols.CORRESPONDING_CHARGING_CAPACITY if output_capacity else CapacityCols.SOH,
                CapacityCols.REMAINING_TIME_TO_CELL_END
            ]
            train_x, train_y = self.__get_multiple_timesteps_soh(
                self.train_discharge_cyc, self.train_discharge_cap, y_indices, steps, multiple_output
            )
            test_x, test_y = self.__get_multiple_timesteps_soh(
                self.test_discharge_cyc, self.test_discharge_cap, y_indices, steps, multiple_output
            )
        else:
            y_indices = [
                CycleCols.DISCHARGING_CAPACITY if output_capacity else CycleCols.SOC,
                CycleCols.REMAINING_TIME_TO_CYCLE_END
            ]
            train_x, train_y = self.__get_multiple_timesteps_soc(
                self.train_discharge_cyc, y_indices, steps, multiple_output
            )
            test_x, test_y = self.__get_multiple_timesteps_soc(
                self.test_discharge_cyc, y_indices, steps, multiple_output
            )

        train_x = self.__scale_multiple_timestep(train_x)
        test_x = self.__scale_multiple_timestep(test_x)

        self.logger.info("Train x: %s, train y: %s | Test x: %s, test y: %s" %
                         (train_x.shape, train_y.shape, test_x.shape, test_y.shape))

        return (train_x, train_y, test_x, test_y)

    def __get_multiple_timesteps_soc(self, cyc, y_indices, steps, multiple_output):
        all_x, all_y = [], []
        for cycle in cyc:
            x, y = self.__cycle_to_multiple_steps_soc(
                y_indices, steps, multiple_output, cycle)
            all_x.append(np.array(x))
            all_y.append(np.array(y))
        all_x = np.concatenate(np.array(all_x)).astype('float32')
        all_y = np.concatenate(np.array(all_y)).astype('float32')
        return all_x, all_y

    def __cycle_to_multiple_steps_soc(self, y_indices, steps, multiple_output, cycle, x_indices=None):
        if(x_indices is None):
            x_indices = self.x_indices
        x, y = [], []
        for i in range(cycle.shape[0]):
            start_ix = i - steps + 1
            x_seq, y_seq = [], []
            y_seq = cycle[i, y_indices]
            # start index is negative, pad zeros to the sequence
            if(start_ix < 0):
                x_seq = np.zeros((abs(start_ix), len(x_indices)))
                x_seq = np.append(
                    x_seq, cycle[0:i+1, x_indices], axis=0)
                if(multiple_output):
                    y_seq = np.zeros((abs(start_ix), len(y_indices)))
                    y_seq = np.append(
                        y_seq, cycle[0:i+1, y_indices], axis=0)
            else:
                x_seq = cycle[start_ix:i+1, x_indices]
                if(multiple_output):
                    y_seq = cycle[start_ix:i+1, y_indices]
            x.append(x_seq)
            y.append(y_seq)
        return x, y

    def __get_multiple_timesteps_soh(self, cyc, cap, y_indices, steps, multiple_output):
        all_x, all_y = [], []
        for i in range(len(cyc)):
            cycle = cyc[i]
            x, y = [], []
            for j in range(len(cycle)):
                start_ix = j - steps + 1
                x_seq, y_seq = [], []
                y_seq = cap[i, y_indices]
                # start index is negative, pad zeros to the sequence
                if(start_ix < 0):
                    x_seq = np.zeros((abs(start_ix), len(self.x_indices)))
                    x_seq = np.append(
                        x_seq, cycle[0:j+1, self.x_indices], axis=0)
                else:
                    x_seq = cycle[start_ix:j+1, self.x_indices]

                if(multiple_output):
                    y_seq = np.repeat([y_seq], steps, axis=0)

                x.append(x_seq)
                y.append(y_seq)
            all_x.append(np.array(x))
            all_y.append(np.array(y))
        all_x = np.concatenate(np.array(all_x)).astype('float32')
        all_y = np.concatenate(np.array(all_y)).astype('float32')
        return all_x, all_y

    def __scale_multiple_timestep(self, x):
        for i in range(len(self.discharge_scalers)):
            # Reshape to 2d array for scaling and reshape back to 3d
            two_d_x = x[:, :, [i]].reshape(x.shape[0]*x.shape[1], 1)
            scaled_two_d_x = self.discharge_scalers[i].transform(two_d_x)
            x[:, :, [i]] = scaled_two_d_x.reshape((x.shape[0], x.shape[1], 1))
        return x

    def get_discharge_grouped_multiple_steps(
        self,
        steps,
        output_capacity=False,
        multiple_output=False
    ):
        train_x, train_y, test_x, test_y = self.get_discharge_whole_cycle(
            output_capacity=output_capacity,
            # always get multiple output form whole cycle, so that we can split them to multiple steps
            multiple_output=True,
            soh=False)

        self.logger.info("Spliting whole cycles to multiple steps...")

        train_x, train_y = self.__whole_cycle_to_multiple_step(
            steps, multiple_output, train_x, train_y)
        test_x, test_y = self.__whole_cycle_to_multiple_step(
            steps, multiple_output, test_x, test_y)

        self.logger.info("Train x: %s, train y: %s | Test x: %s, test y: %s" %
                         (train_x.shape, train_y.shape, test_x.shape, test_y.shape))

        return (train_x, train_y, test_x, test_y)

    def __whole_cycle_to_multiple_step(self, steps, multiple_output, whole_cycle_x, whole_cycle_y):
        x_indices = np.arange(whole_cycle_x.shape[-1])
        y_indices = np.arange(whole_cycle_y.shape[-1])
        new_x = []
        new_y = []
        for i in range(len(whole_cycle_x)):
            x, y = self.__cycle_to_multiple_steps(
                y_indices, steps, multiple_output, whole_cycle_x[i], whole_cycle_y[i], x_indices)
            new_x.append(x)
            new_y.append(y)
        whole_cycle_x = np.array(new_x)
        whole_cycle_y = np.array(new_y)
        return whole_cycle_x, whole_cycle_y

    def __cycle_to_multiple_steps(self, y_indices, steps, multiple_output, cycle_x, cycle_y, x_indices=None):
        if(x_indices is None):
            x_indices = self.x_indices
        x, y = [], []
        for i in range(cycle_x.shape[0]):
            start_ix = i - steps + 1
            x_seq, y_seq = [], []
            y_seq = cycle_y[i, y_indices]
            # start index is negative, pad zeros to the sequence
            if(start_ix < 0):
                x_seq = np.zeros((abs(start_ix), len(x_indices)))
                x_seq = np.append(
                    x_seq, cycle_x[0:i+1, x_indices], axis=0)
                if(multiple_output):
                    y_seq = np.zeros((abs(start_ix), len(y_indices)))
                    y_seq = np.append(
                        y_seq, cycle_y[0:i+1, y_indices], axis=0)
            else:
                x_seq = cycle_x[start_ix:i+1, x_indices]
                if(multiple_output):
                    y_seq = cycle_y[start_ix:i+1, y_indices]
            x.append(x_seq)
            y.append(y_seq)
        return x, y

    def keep_only_capacity(self, y, is_multiple_output=False, is_grouped_multiple_step=False):
        if is_grouped_multiple_step:
            if is_multiple_output:
                new_y = y[:, :, :, 0]
            else:
                new_y = y[:, :, 0]
        else:
            if is_multiple_output:
                new_y = y[:, :, 0]
            else:
                new_y = y[:, 0]
        self.logger.info("New y: %s" % (new_y.shape,))
        return new_y

    def keep_only_time(self, y, is_multiple_output=False, is_grouped_multiple_step=False):
        if is_grouped_multiple_step:
            if is_multiple_output:
                new_y = y[:, :, :, 1]
            else:
                new_y = y[:, :, 1]
        else:
            if is_multiple_output:
                new_y = y[:, :, 1]
            else:
                new_y = y[:, 1]
        self.logger.info("New y: %s" % (new_y.shape,))
        return new_y

    def get_discharge_whole_cycle_future(self, train_names, test_names, min_cycle_length=None):
        y_indices = [CapacityCols.CORRESPONDING_CHARGING_CAPACITY]
        train_raw_x, train_y = self.__get_whole_cycle_soh_x_y(
            self.train_discharge_cyc, self.train_discharge_cap, self.x_indices, y_indices
        )
        test_raw_x, test_y = self.__get_whole_cycle_soh_x_y(
            self.test_discharge_cyc, self.test_discharge_cap, self.x_indices, y_indices
        )

        train_x, test_x = self.__get_padded_whole_cycle(train_raw_x, test_raw_x, min_cycle_length)
        train_x[:,:,1] = np.negative(train_x[:, :, 1])
        test_x[:,:,1] = np.negative(test_x[:, :, 1])
        current_train = train_x[:, :, 1]
        current_test = test_x[:, :, 1]


        battery_name_cycle_train = self.train_discharge_cap[:, [CapacityCols.TEST_NAME]]
        battery_range_cycle_train = np.array([np.where(battery_name_cycle_train == x)[0][-1]+1 for x in train_names])
        battery_name_cycle_test = self.test_discharge_cap[:, [CapacityCols.TEST_NAME]]
        battery_range_cycle_test = np.array([np.where(battery_name_cycle_test == x)[0][-1]+1 for x in test_names])


        time_train, _ = self.__get_whole_cycle_soh_x_y(
            self.train_discharge_cyc, self.train_discharge_cap, [CycleCols.STEP_TIME], y_indices
        )
        time_test, _ = self.__get_whole_cycle_soh_x_y(
            self.test_discharge_cyc, self.test_discharge_cap, [CycleCols.STEP_TIME], y_indices
        )
        time_train, time_test = self.__get_padded_whole_cycle(time_train, time_test, min_cycle_length)


        self.logger.info('''Train x: %s, train y soh: %s | Test x: %s, test y soh: %s | 
                            battery n cycle train: %s, battery n cycle test: %s, 
                            time train: %s, time test: %s |
                            raw current train: %s, raw current test: %s |
                            ''' %
                         (train_x.shape, train_y.shape, test_x.shape, test_y.shape,
                          battery_range_cycle_train.shape, battery_range_cycle_test.shape, 
                          time_train.shape, time_test.shape,
                          current_train.shape, current_test.shape))

        return (train_x, train_y, test_x, test_y,
                battery_range_cycle_train, battery_range_cycle_test,
                time_train, time_test,
                current_train, current_test)
