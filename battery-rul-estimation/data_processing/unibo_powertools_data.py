import logging
from pathlib import Path

import numpy as np
import pandas as pd

TEST_RESULT_DATA_PATH = 'data/unibo-powertools-dataset/unibo-powertools-dataset/test_result.csv'
TEST_RESULT_TRIAL_END_DATA_PATH = 'data/unibo-powertools-dataset/unibo-powertools-dataset/test_result_trial_end.csv'

# (test_name, record_id)
ABNORMAL_CYCLE_RECORDS = [
    ('006-EE-2.85-0820-S', 621391),  # Discharge capacity dropped abnormally
    ('006-EE-2.85-0820-S', 621392)  # Discharge capacity dropped abnormally
]
ABNORMAL_CAPACITY_RECRODS = [
    ('007-EE-2.85-0820-S', 623002)  # Not the last row in cycle
]


class CycleCols:
    TEST_NAME = 0
    RECORD_ID = 1
    TIME = 2
    STEP_TIME = 3
    LINE = 4
    VOLTAGE = 5
    CURRENT = 6
    CHARGING_CAPACITY = 7
    DISCHARGING_CAPACITY = 8
    WH_CHARGING = 9
    WH_DISCHARGING = 10
    TEMPERATURE = 11
    CYCLE_COUNT = 12
    SOC = 13
    REMAINING_TIME_TO_CYCLE_END = 14


class CapacityCols:
    TEST_NAME = 0
    RECORD_ID = 1
    TIME = 2
    STEP_TIME = 3
    LINE = 4
    VOLTAGE = 5
    CURRENT = 6
    CHARGING_CAPACITY = 7
    DISCHARGING_CAPACITY = 8
    WH_CHARGING = 9
    WH_DISCHARGING = 10
    TEMPERATURE = 11
    CYCLE_COUNT = 12
    MAX_TEMPERATURE = 13
    AVERAGE_TENSION = 14
    REMAINING_TIME_TO_CELL_END = 15
    MAXIMUM_CAPACITY = 16
    NOMINAL_CAPACITY = 17
    SOH = 18
    CORRESPONDING_CHARGING_CAPACITY = 19


class UniboPowertoolsData():
    def __init__(self,
                 test_types=[],
                 chunk_size=1000000,
                 lines=[37, 40],
                 charge_line=37,
                 discharge_line=40,
                 base_path="./"):

        self.logger = logging.getLogger()
        self.test_types = test_types if test_types is not None else []
        self.chunksize = chunk_size
        self.lines = lines if lines is not None else []
        self.charge_line = charge_line
        self.discharge_line = discharge_line
        self.cyc_path = base_path + TEST_RESULT_DATA_PATH
        self.cap_path = base_path + TEST_RESULT_TRIAL_END_DATA_PATH

        self.__load_raw_data()

    def __load_raw_data(self):
        self.__load_csv_to_raw()
        self.__clean_cycle_raw()
        self.__clean_capacity_raw()
        self.__assign_charge_raw()
        self.__assign_discharge_raw()

    def __load_csv_to_raw(self):
        self.logger.debug("Start loading data with lines: %s, types: %s and chunksize: %s..." %
                          (self.lines, self.test_types, self.chunksize))

        iter_cyc = pd.read_csv(
            self.cyc_path, chunksize=self.chunksize, iterator=True)
        self.cycle_raw = pd.concat(self.__filter_raw_chunk(iter_cyc))

        iter_cap = pd.read_csv(
            self.cap_path, chunksize=self.chunksize, iterator=True)
        self.cap_raw = pd.concat(self.__filter_raw_chunk(iter_cap))

        self.logger.debug("Finish loading data.")
        self.logger.info("Loaded raw dataset A data with cycle row count: %s and capacity row count: %s" %
                         (len(self.cycle_raw), len(self.cap_raw)))

    def __filter_raw_chunk(self, iter_chunk):
        # Collect all conditions first.
        # If no test name and lines specified, get all data from chunk without filtering
        conditions = list()
        if(len(self.test_types) > 0):
            conditions.append(
                'test_name.str.endswith(tuple(%s))' % self.test_types)
        if(len(self.lines) > 0):
            conditions.append('line.isin(%s)' % self.lines)

        filter_cks = []
        for chunk in iter_chunk:
            if(len(conditions) > 0):
                chunk = chunk.query('&'.join(conditions), engine='python')
            filter_cks.append(chunk)
        return filter_cks

    def __clean_cycle_raw(self):
        self.logger.debug("Start cleaning cycle raw data...")
        count_before = len(self.cycle_raw)

        # Voltage outside 0.1 ~ 5.0 are seen as abnormal dataset
        self.cycle_raw = self.cycle_raw.drop(
            self.cycle_raw[(self.cycle_raw['voltage'] > 5.0)
                           | (self.cycle_raw['voltage'] < 0.1)].index)

        # Filter all predefined abnormal records
        self.cycle_raw = self.__filter_predefined(
            self.cycle_raw, ABNORMAL_CYCLE_RECORDS)

        self.logger.debug("Finish cleaning cycle raw data.")
        self.logger.info("Removed %s rows of abnormal cycle raw data." %
                         (count_before - len(self.cycle_raw)))

    def __filter_predefined(self, raw_data, predefined_records):
        for abn_record in predefined_records:
            raw_data.drop(
                raw_data[(raw_data['test_name'] == abn_record[0])
                         & (raw_data['record_id'] == abn_record[1])].index, inplace=True)
        return raw_data

    def __clean_capacity_raw(self):
        self.logger.debug("Start cleaning capacity raw data...")
        count_before = len(self.cap_raw)

        # Filter all predefined abnormal records
        self.cap_raw = self.__filter_predefined(
            self.cap_raw, ABNORMAL_CAPACITY_RECRODS)

        self.logger.debug("Finish cleaning capacity raw data.")
        self.logger.info("Removed %s rows of abnormal capacity raw data." %
                         (count_before - len(self.cap_raw)))

    def __assign_charge_raw(self):
        self.logger.debug("Start assigning charging raw data...")

        self.charge_cyc_raw = self.cycle_raw[self.cycle_raw['line']
                                             == self.charge_line]
        self.charge_cap_raw = self.cap_raw[self.cap_raw['line']
                                           == self.charge_line]

        self.logger.debug("Finish assigning charging raw data.")
        self.logger.info("[Charging] cycle raw count: %s, capacity raw count: %s"
                         % (len(self.charge_cyc_raw), len(self.charge_cap_raw)))

    def __assign_discharge_raw(self):
        self.logger.debug("Start assigning discharging raw data...")

        self.discharge_cyc_raw = self.cycle_raw[self.cycle_raw['line']
                                                == self.discharge_line]
        self.discharge_cap_raw = self.cap_raw[self.cap_raw['line']
                                              == self.discharge_line]

        self.logger.debug("Finish assigning discharging raw data.")
        self.logger.info("[Discharging] cycle raw count: %s, capacity raw count: %s"
                         % (len(self.discharge_cyc_raw), len(self.discharge_cap_raw)))

    def prepare_data(self, train_names, test_names):
        self.logger.debug("Start preparing data for training: %s and testing: %s..."
                          % (train_names, test_names))

        self.train_charge_cyc, self.train_charge_cap = self.__get_cyc_and_cap(
            train_names, self.charge_cyc_raw, self.charge_cap_raw
        )
        self.test_charge_cyc, self.test_charge_cap = self.__get_cyc_and_cap(
            test_names, self.charge_cyc_raw, self.charge_cap_raw
        )
        self.logger.debug("Finish getting training and testing charge data.")

        self.train_discharge_cyc, self.train_discharge_cap = self.__get_cyc_and_cap(
            train_names, self.discharge_cyc_raw, self.discharge_cap_raw
        )
        self.test_discharge_cyc, self.test_discharge_cap = self.__get_cyc_and_cap(
            test_names, self.discharge_cyc_raw, self.discharge_cap_raw
        )
        self.logger.debug(
            "Finish getting training and testing discharge data.")

        self.train_charge_cyc, self.train_charge_cap = self.__clean_cyc_and_cap_without_mapping(
            self.train_charge_cyc, self.train_charge_cap, self.train_discharge_cap
        )
        self.test_charge_cyc, self.test_charge_cap = self.__clean_cyc_and_cap_without_mapping(
            self.test_charge_cyc, self.test_charge_cap, self.test_discharge_cap
        )
        self.logger.debug("Finish cleaning training and testing charge data.")

        self.train_discharge_cyc, self.train_discharge_cap = self.__clean_cyc_and_cap_without_mapping(
            self.train_discharge_cyc, self.train_discharge_cap, self.train_charge_cap
        )
        self.test_discharge_cyc, self.test_discharge_cap = self.__clean_cyc_and_cap_without_mapping(
            self.test_discharge_cyc, self.test_discharge_cap, self.test_charge_cap
        )
        self.logger.debug(
            "Finish cleaning training and testing discharge data.")

        self.train_discharge_cyc = self.__add_discharge_soc_pars(
            self.train_discharge_cyc, self.train_charge_cap
        )
        self.test_discharge_cyc = self.__add_discharge_soc_pars(
            self.test_discharge_cyc, self.test_charge_cap
        )
        self.logger.debug(
            "Finish adding training and testing discharge SOC parameters.")

        self.train_discharge_cap = self.__add_discharge_soh_pars(
            self.train_discharge_cap, self.train_charge_cap)
        self.test_discharge_cap = self.__add_discharge_soh_pars(
            self.test_discharge_cap, self.test_charge_cap)
        self.logger.debug(
            "Finish adding training and testing discharge SOH parameters.")

        self.logger.debug("Finish preparing data.")
        self.logger.info("Prepared training charge cycle data: %s, capacity data: %s" %
                         (self.train_charge_cyc.shape, self.train_charge_cap.shape))
        self.logger.info("Prepared testing charge cycle data: %s, capacity data: %s" %
                         (self.test_charge_cyc.shape, self.test_charge_cap.shape))
        self.logger.info("Prepared training discharge cycle data: %s, capacity data: %s" %
                         (self.train_discharge_cyc.shape, self.train_discharge_cap.shape))
        self.logger.info("Prepared testing discharge cycle data: %s, capacity data: %s" %
                         (self.test_discharge_cyc.shape, self.test_discharge_cap.shape))

    def __get_cyc_and_cap(self, names, cyc_raw, cap_raw):
        cyc_data = []
        cap_data = []

        gp_cyc_raw = self.__group_cyc_by_name(cyc_raw, names)

        gp_cap_raw = cap_raw.groupby('test_name')

        for test_name in names:
            last_cap_group_index = 0
            cap_group = gp_cap_raw.get_group(
                test_name).reset_index(drop=True)

            for cycle in gp_cyc_raw[test_name]:
                cycle = cycle.reset_index(drop=True)
                cycle_count = cycle.iloc[-1]['cycle_count']

                target_cap_row_indices = np.array(
                    cap_group.index[(cap_group['cycle_count'] == cycle_count)])
                target_cap_row_index = target_cap_row_indices[
                    target_cap_row_indices >= last_cap_group_index][0]
                target_cap_row = cap_group.iloc[target_cap_row_index]

                last_cap_group_index = target_cap_row_index

                cyc_data.append(cycle.values)
                cap_data.append(target_cap_row.values)

        cyc_data = np.array(cyc_data, dtype=object)
        cap_data = np.array(cap_data, dtype=object)

        return (cyc_data, cap_data)

    def __group_cyc_by_name_and_cyc_count(self, cyc_raw):
        return cyc_raw.groupby(
            ['test_name', (cyc_raw['cycle_count'] !=
                           cyc_raw['cycle_count'].shift()).cumsum()]
        )

    def __group_cyc_by_name(self, cyc_raw, test_names):
        grouped_cycle = self.__group_cyc_by_name_and_cyc_count(cyc_raw)
        grouped_name_cycle = {}
        for key, group in grouped_cycle:
            test_name = key[0]
            if(test_name not in grouped_name_cycle):
                grouped_name_cycle[test_name] = []
            grouped_name_cycle[test_name].append(group)
        return grouped_name_cycle

    def __clean_cyc_and_cap_without_mapping(self, target_cyc, target_cap, mapping_cap):
        """Clean all charge/discharge cycle which does not have corresponding mapping discharge/charge cycle"""
        clean_indices = []
        dirty_row = 0
        for i in range(len(target_cyc)):
            if(i >= len(mapping_cap) or
               target_cap[i][CapacityCols.CYCLE_COUNT] != mapping_cap[i-dirty_row][CapacityCols.CYCLE_COUNT]):
                dirty_row += 1
            else:
                clean_indices.append(i)

        return (target_cyc[clean_indices], target_cap[clean_indices])

    def __add_discharge_soc_pars(self, discharge_cyc, charge_cap):
        for i in range(len(discharge_cyc)):
            # SOC: (last charge cycle capacity - discharging capacity) / last charge cycle capacity
            discharge_cyc[i] = np.c_[discharge_cyc[i],
                                     np.zeros(discharge_cyc[i].shape[0])]
            discharge_cyc[i][:, -1] = (charge_cap[i][CapacityCols.CHARGING_CAPACITY] - discharge_cyc[i]
                                       [:, CycleCols.DISCHARGING_CAPACITY]) / charge_cap[i][CapacityCols.CHARGING_CAPACITY]

            # Time remaining to cycle end: (Time of last row in cycle - current time)
            discharge_cyc[i] = np.c_[discharge_cyc[i],
                                     np.zeros(discharge_cyc[i].shape[0])]
            discharge_cyc[i][:, -1] = discharge_cyc[i][-1:,
                                                       CapacityCols.TIME] - discharge_cyc[i][:, CycleCols.TIME]

        return discharge_cyc

    def __add_discharge_soh_pars(self, discharge_cap, charge_cap):
        discharge_cap = np.c_[discharge_cap,
                              np.zeros((discharge_cap.shape[0], 5))]

        for cap in discharge_cap:
            # Time remaining to cell end: (Time of last row in the cell - current time)
            cap[CapacityCols.REMAINING_TIME_TO_CELL_END] = discharge_cap[discharge_cap[:, CapacityCols.TEST_NAME] ==
                                                                         cap[CapacityCols.TEST_NAME]][-1][CapacityCols.TIME] - cap[CapacityCols.TIME]

            # Maximum capacity in corresponding charging cycles
            same_test_charge_cap = charge_cap[charge_cap[:,
                                                         CapacityCols.TEST_NAME] == cap[CapacityCols.TEST_NAME]]
            cap[CapacityCols.MAXIMUM_CAPACITY] = np.max(
                same_test_charge_cap[:, CapacityCols.CHARGING_CAPACITY])

            # Nominal cell capacity
            cell_cap = cap[CapacityCols.MAXIMUM_CAPACITY]
            # Test name convention: 000-XW-Y.Y-AABB-T (7~10 chars are cell capacity)
            cell_cap_text = cap[CapacityCols.TEST_NAME][7:10]
            try:
                cell_cap = float(cell_cap_text)
            except Exception:
                pass
            cap[CapacityCols.NOMINAL_CAPACITY] = cell_cap

        # SOH: (Last charging cycle capacity / nominal cell capacity)
        discharge_cap[:, CapacityCols.SOH] = charge_cap[:,
                                                        CapacityCols.CHARGING_CAPACITY] / discharge_cap[:, CapacityCols.MAXIMUM_CAPACITY]

        # Corresponding charging cycle charging capacity
        discharge_cap[:, CapacityCols.CORRESPONDING_CHARGING_CAPACITY] = charge_cap[:,
                                                                                    CapacityCols.CHARGING_CAPACITY]

        return discharge_cap

    def get_charge_data(self):
        return (
            self.train_charge_cyc,
            self.train_charge_cap,
            self.test_charge_cyc,
            self.test_charge_cap
        )

    def get_discharge_data(self):
        return (
            self.train_discharge_cyc,
            self.train_discharge_cap,
            self.test_discharge_cyc,
            self.test_discharge_cap
        )

    def get_all_test_names(self):
        return self.cycle_raw['test_name'].unique()
