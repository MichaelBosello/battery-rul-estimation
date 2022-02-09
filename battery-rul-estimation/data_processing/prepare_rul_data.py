import numpy as np
import logging


class RulHandler():
    def __init__(self):
        self.logger = logging.getLogger()

    def prepare_y_future(self, battery_names, battery_n_cycle, y_soh, current, time, capacity_threshold=None, allow_negative_future = False, capacity=None):
        cycle_lenght = current.shape[1]
        battery_range_step = [x * cycle_lenght for x in battery_n_cycle]
        self.logger.info("battery step: {}".format(battery_n_cycle))
        self.logger.info("battery ranges: {}".format(battery_range_step))

        if capacity is None:
            battery_nominal_capacity = [ float(name.split("-")[2]) for name in battery_names]
        else:
            battery_nominal_capacity = [ capacity for name in battery_names ]

        current = current.ravel()
        time = time.ravel()
        capacity_integral_train = []
        a = 0
        for battery_index, b in enumerate(battery_range_step):
            self.logger.info("processing range {} - {}".format(a, b))
            integral_sum = 0
            pre_i = a
            for i in range(a,b,cycle_lenght):
                integral = np.trapz(
                    y=current[pre_i:i][current[pre_i:i]>0],
                    x=time[pre_i:i][current[pre_i:i]>0])
                integral_sum += integral
                pre_i = i
                capacity_integral_train.append(integral_sum/battery_nominal_capacity[battery_index])
            a = b
        capacity_integral_train = np.array(capacity_integral_train)
        self.logger.info("Train integral: {}".format(capacity_integral_train.shape))

        y_future = []
        a = 0
        for battery_index, b in enumerate(battery_n_cycle):
            self.logger.info("processing range {} - {}".format(a, b))
            if capacity_threshold is None:
                index = b-1
            else:
                index = np.argmax(y_soh[a:b]<capacity_threshold[battery_nominal_capacity[battery_index]]) + a
                if index == a:
                    index = b-1
            self.logger.info("threshold index: {}".format(index))
            for i in range(a, b):
                if not allow_negative_future:
                    y = capacity_integral_train[index] - capacity_integral_train[i] if i < index else 0
                else:
                    y = capacity_integral_train[index] - capacity_integral_train[i]
                y_future.append(y)
            a = b
        y_future = np.array(y_future)
        self.logger.info("y future: {}".format(y_future.shape))

        y_with_future = np.column_stack((capacity_integral_train, y_future))
        self.logger.info("y with future: {}".format(y_with_future.shape))
        return y_with_future

    def compress_cycle(self, train_x, test_x):
        train_x[train_x == 0] = np.nan
        new_train = np.empty((train_x.shape[0], train_x.shape[2], 2))
        for i in range(train_x.shape[2]):
            for x in range(train_x.shape[0]):#full matrix takes too much RAM for in place mean/std
                new_train[x,i,0] = np.nanmean(train_x[x,:,i])
                new_train[x,i,1] = np.nanstd(train_x[x,:,i])
        new_train = new_train.reshape((train_x.shape[0], train_x.shape[2]*2))

        test_x[test_x == 0] = np.nan
        new_test = np.empty((test_x.shape[0], test_x.shape[2], 2))
        for i in range(test_x.shape[2]):
            for x in range(test_x.shape[0]):#full matrix takes too much RAM for in place mean/std
                new_test[x,i,0] = np.nanmean(test_x[x,:,i])
                new_test[x,i,1] = np.nanstd(test_x[x,:,i])
        new_test = new_test.reshape((test_x.shape[0], test_x.shape[2]*2))

        self.logger.info("new compact train x: {}, new compact test x: {}".format(new_train.shape, new_test.shape))
        return new_train, new_test

    def battery_life_to_time_series(self, x, n_cycle, battery_range_cycle):
        series = np.zeros((x.shape[0], n_cycle, x.shape[1]))
        a = 0
        for b in battery_range_cycle:
            for i in range(a,b):
                bounded_a = max(a, i+1-n_cycle)
                series[i,0:i+1-bounded_a] = x[bounded_a:i+1]
            a = b
        self.logger.info("x time serie shape: {}".format(series.shape))
        return series

    def delete_initial(self, x, y, battery_range, soh, warmup):
        new_range = [x - warmup*(i+1) for i, x in enumerate(battery_range)]
        battery_range = np.insert(battery_range[:-1], 0, [0]) 
        indexes = [int(x+i) for x in battery_range for i in range(warmup)]
        x = np.delete(x, indexes, axis=0)
        y = np.delete(y, indexes, axis=0)
        soh = np.delete(soh.flatten(), indexes, axis=0)
        self.logger.info("x with warmup: {}, y with warmup: {}".format(x.shape, y.shape))
        return x, y, new_range, soh

    def limit_zeros(self, x, y, battery_range, soh, limit=100):
        indexes = []
        new_range = []
        a = 0
        removed = 0
        for b in battery_range:
            zeros = np.where(y[a:b,1] == 0)[0]
            zeros = zeros + a
            indexes.extend(zeros[limit:].tolist())
            removed = removed + len(zeros[limit:])
            new_range.append(b - removed)
            a = b
        x = np.delete(x, indexes, axis=0)
        y = np.delete(y, indexes, axis=0)
        soh = np.delete(soh.flatten(), indexes, axis=0)
        self.logger.info("x with limit: {}, y with limit: {}".format(x.shape, y.shape))
        return x, y, new_range, soh

    def unify_datasets(self, x, y, battery_range, soh, m_x, m_y, m_battery_range, m_soh):
        m_battery_range = m_battery_range + battery_range[-1]
        x = np.concatenate((x, m_x))
        y = np.concatenate((y, m_y))
        battery_range = np.concatenate((battery_range, m_battery_range))
        soh = np.concatenate((soh.flatten(), m_soh))

        self.logger.info('''Unified x: %s, unified y : %s, unified battery range: %s''' %
                         (x.shape, y.shape, battery_range))

        return (x, y, battery_range, soh)

    class Normalization():
        def fit(self, train):
            if len(train.shape) == 1:
                self.case = 1
                self.min = min(train)
                self.max = max(train)
            elif len(train.shape) == 2:
                self.case = 2
                self.min = [min(train[:,i]) for i in range(train.shape[1])]
                self.max = [max(train[:,i]) for i in range(train.shape[1])]
            elif len(train.shape) == 3:
                self.case = 3
                self.min = [train[:,:,i].min() for i in range(train.shape[2])]
                self.max = [train[:,:,i].max() for i in range(train.shape[2])]

        def normalize(self, data):
            if self.case == 1:
                data = (data - self.min) / (self.max - self.min)
            elif self.case == 2:
                for i in range(data.shape[1]):
                    data[:,i] = (data[:,i] - self.min[i]) / (self.max[i] - self.min[i])
            elif self.case == 3:
                for i in range(data.shape[2]):
                    data[:,:,i] = (data[:,:,i] - self.min[i]) / (self.max[i] - self.min[i])
            return data

        def fit_and_normalize(self, train, test, val=None):
            self.fit(train)
            if val is not None:
                return self.normalize(train), self.normalize(test), self.normalize(val)
            else:
                return self.normalize(train), self.normalize(test)


        def denormalize(self, a):
            if self.case == 1:
                a = a * (self.max - self.min) + self.min
            elif self.case == 2:
                for i in range(a.shape[1]):
                    a[:,i] = a[:,i] * (self.max[i] - self.min[i]) + self.min[i]
            elif self.case == 3:
                for i in range(a.shape[2]):
                    a[:,:,i] = a[:,:,i] * (self.max[i] - self.min[i]) + self.min[i]
            return a