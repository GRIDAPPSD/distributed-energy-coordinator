# Copyright (C) 2017-2019 Battelle Memorial Institute
# file: HEMS_agent.py
"""GridAPPS-D: Distributed Energy Coordinator Application
    @author: monish_mukherjee

Public Functions:
    :HEMS_agent: initializes and runs an instance of the HEMS_agent

"""

import numpy as np
import SPARQL_query as query_cim
import matplotlib.pyplot as plt
from scipy import interpolate
from pulp import *
import logging
import os

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

class HEMS:
    """This agent determines optimial schedules for customers with batteries as DERs

    Args:
        TODO: update inputs for this agent

"""
    def __init__(self, name, bus, optimization=False): #TODO: update inputs for class
        """Initializes the class
        """
        # TODO: update attributes of class
        # initialize from Args:
        self.name = name
        self.feeder_mrid = "_59AD8E07-3BF9-A4E2-CB8F-C3722F837B62"
        self.bus_name = bus
        self.load = query_cim.get_load_info_from_bus(self.feeder_mrid, self.bus_name)
        for load_name in self.load:
            self.load[load_name]['set_point'] = self.load[load_name]['p']
            self.load[load_name]['forecast'] = np.array(0)

        self.PV = query_cim.get_PV_info_from_bus(self.feeder_mrid, self.bus_name)
        for PV_name in self.PV:
            self.PV[PV_name]['set_point'] = 0.0
            self.PV[PV_name]['forecast'] = np.array(0)

        self.BESS = query_cim.get_BESS_info_from_bus(self.feeder_mrid, self.bus_name)
        for BESS_name in self.BESS:
            self.BESS[BESS_name]['SOC'] = self.BESS[BESS_name]['storedE']/self.BESS[BESS_name]['ratedE']
            self.BESS[BESS_name]['set_point'] = 0.0
            self.BESS[BESS_name]['schedule'] = np.array(0)

        logger.info("HEMS Agent {}: Initialized".format(self.name))

    def get_price_forecast(self):
        price_file_path =  "../inputs/price.csv"
        self.price_forecast = np.genfromtxt(price_file_path, delimiter=',')

        logger.info("HEMS Agent {}: Getting price info".format(self.name))

    def get_load_forecast(self, interval, duration):
        load_shape_file_path =  "../inputs/loadshape1.csv"
        load_shape = np.genfromtxt(load_shape_file_path, delimiter=',')
        for load in self.load:
            load_kW = self.load[load]['p']*load_shape
            load_interp = interpolate.interp1d(np.linspace(0, duration - 1, len(load_kW)), load_kW)
            load_kW_intv = load_interp(np.linspace(0, duration - 1, duration))
            self.load[load]['forecast'] = load_kW_intv

        logger.info("HEMS Agent {}: Getting load forecasts".format(self.name))

    def get_solar_forecast(self, interval, duration):
        solar_shape_file_path =  "../inputs/pvshape.csv"
        solar_shape = np.genfromtxt(solar_shape_file_path, delimiter=',')
        for pv in self.PV:
            solar_kW = self.PV[pv]['p']*solar_shape
            solar_interp = interpolate.interp1d(np.linspace(0, duration - 1, len(solar_kW)), solar_kW)
            solar_kW_intv = solar_interp(np.linspace(0, duration - 1, duration))
            self.PV[pv]['forecast'] = solar_kW_intv

        logger.info("HEMS Agent {}: Getting solar forecasts".format(self.name))


    def optimize_battery(self, interval=1, duration=24, plot_schedules=False):
        self.get_price_forecast()
        self.get_load_forecast(interval, duration)
        self.get_solar_forecast(interval, duration)

        T = int(duration / interval)
        net_pv = np.zeros((T))
        net_load = np.zeros((T))
        for pv in self.PV:
            net_pv += self.PV[pv]['forecast']
        for load in self.load:
            net_load += self.load[load]['forecast']

        ###########################################################################
        ####################  Initializing Battery Parameters  ####################
        ###########################################################################
        batt_diseff   = []
        batt_cheff    = []
        batt_es       = []
        batt_dispower = []
        batt_chpower  = []
        batt_soc0     = []
        batt_socmax   = []
        batt_socmin   = []

        batt_diseff_default = 0.9
        batt_cheff_default = 0.9
        batt_socmax_default = 1.0
        batt_socmin_default = 0.2

        TIME = range(0, T)
        Battery_Pch = {}
        Battery_Pdisc = {}
        Battery_state = {}

        for i, batt in enumerate(self.BESS):
            batt_diseff.append(batt_diseff_default)
            batt_cheff.append(batt_cheff_default)
            batt_socmax.append(batt_socmax_default)
            batt_socmin.append(batt_socmin_default)

            batt_es.append(self.BESS[batt]['ratedE'])
            batt_dispower.append(self.BESS[batt]['ratedS'])
            batt_chpower.append(self.BESS[batt]['ratedS'])
            batt_soc0.append(self.BESS[batt]['storedE']/self.BESS[batt]['ratedE'])

        ###########################################################################
        ########################## Decision variables #############################
        ###########################################################################
        prob_DER = LpProblem("Batt_optimize", LpMinimize)
        P_customer = LpVariable.dicts("Customer", TIME, -10000, 10000, cat='Continuous')
        for i, batt in enumerate(self.BESS):
            Battery_Pch[i] = []
            Battery_Pch[i] = LpVariable.dicts("BatteryCh" + str(i), TIME, 0, batt_chpower[i],cat='Continuous')
            Battery_Pdisc[i] = []
            Battery_Pdisc[i] = LpVariable.dicts("BatteryDisch" + str(i), TIME, 0, batt_dispower[i], cat='Continuous')
            # Battery_state[i] = []
            # Battery_state[i] = LpVariable.dicts("BatterySt" + str(i), TIME, lowBound=0, upBound=1, cat='Binary')

        prob_DER += lpSum((self.price_forecast[t] * P_customer[t] /1000) for t in TIME)

        ###########################################################################
        ########################  Inequality Constraints ##########################
        ###########################################################################
        for m in range(len(self.BESS)):
            for t in range(1, T):
                prob_DER += lpSum(Battery_Pdisc[m][k] * (1 / batt_diseff[m]) - Battery_Pch[m][k] * batt_cheff[m] for k in range(t)) * interval <= batt_soc0[m] * batt_es[m]
                prob_DER += lpSum(Battery_Pdisc[m][k] * (-1 / batt_diseff[m]) + Battery_Pch[m][k] * batt_cheff[m] for k in range(t)) * interval <= batt_es[m] * batt_socmax[m] - batt_soc0[m] * batt_es[m]
                prob_DER += lpSum(Battery_Pdisc[m][k] * (-1 / batt_diseff[m]) + Battery_Pch[m][k] * batt_cheff[m] for k in range(t)) * interval >= batt_es[m] * batt_socmin[m] - batt_soc0[m] * batt_es[m]

        ##### Extra constraint to avoid simualatenous charging & discharging ######
        # for m in range(no_batt):
        #     for t in range(1,T):
        #         prob_DER +=   Battery_Pdisc[m][t]*(1/batt_dispower[m]) + Battery_state[m][t] <= 1
        #         prob_DER +=   -1*Battery_Pch[m][t]*(1/batt_chpower[m]) + Battery_state[m][t] >= 0

        ###########################################################################
        ########################  Equality Constraints ############################
        ###########################################################################
        for t in range(T):
            prob_DER += lpSum(Battery_Pdisc[n][t] - Battery_Pch[n][t] for n in range(len(self.BESS))) + P_customer[t] == net_load[t] - net_pv[t]

        for m in range(len(self.BESS)):
            prob_DER += lpSum((Battery_Pdisc[m][t] * (1 / batt_diseff[m]) - Battery_Pch[m][t] * batt_cheff[m]) for t in range(T)) == 0

        ################# Extra constraint for identical battery behavior #########
        # for t in range(T):
        #     prob_DER +=  (Battery_Pdisc[0][t] - Battery_Pdisc[1][t]) == 0
        #     prob_DER +=  (Battery_Pch[0][t] - Battery_Pch[1][t]) == 0

        logger.info("HEMS Agent {}: BESS finding optimal BESS schedules".format(self.name))
        prob_DER.writeLP("Batt_opt.lp")
        prob_DER.solve(PULP_CBC_CMD(msg=False))
        # prob_EWH.solve(options=['set mip tolerances mipgap 0.0025'])
        logger.info("HEMS Agent {}: BESS schedules status -- {}".format(self.name, LpStatus[prob_DER.status]))

        ###########################################################################
        ############ Reading Outputs from Optimization Constraints ################
        ###########################################################################
        Battery_disch_output = np.zeros((len(self.BESS), T))
        Battery_ch_output = np.zeros((len(self.BESS), T))
        P_customer = np.zeros(T)
        for v in prob_DER.variables():
            var_name = v.name
            time = int(var_name.split('_')[1])
            if 'BatteryCh' in var_name:
                batt_idx = int(var_name.split('_')[0][-1])
                Battery_ch_output[batt_idx, time] = v.varValue
            elif 'BatteryDisch' in var_name:
                batt_idx = int(var_name.split('_')[0][-1])
                Battery_disch_output[batt_idx, time] = v.varValue
            elif 'Customer' in var_name:
                P_customer[time] = v.varValue

        # for i in range(len(self.BESS)):
        #     print('Battery' + str(i) + 'product', Battery_ch_output[i, :] * Battery_disch_output[i, :])

        Battery_power = np.zeros((len(self.BESS), T))
        Battery_SOC = np.zeros((len(self.BESS), T + 1))

        for i, batt in enumerate(self.BESS):
            Battery_power[i, :] = Battery_disch_output[i, :] - Battery_ch_output[i, :]
            self.BESS[batt]['schedule'] = Battery_power[i, :]

            pbatt_ch = -(Battery_power[i, :] - np.abs(Battery_power[i, :])) / 2 * batt_cheff[i]
            pbatt_disch = (Battery_power[i, :] + np.abs(Battery_power[i, :])) / 2 / batt_diseff[i]
            Battery_SOC[i, 0] = batt_soc0[i]
            for t in range(len(Battery_power[i, :])):
                Battery_SOC[i, t + 1] = Battery_SOC[i, t] + (pbatt_ch[t] / batt_es[i] - pbatt_disch[t] / batt_es[i]) * interval

        ###########################################################################
        ####################### Plotting Optimization Results #####################
        ###########################################################################
        if plot_schedules:
            logger.info("HEMS Agent {}: Plotting optimization results".format(self.name))
            fig, ax = plt.subplots(1, 1, figsize=(16, 7), dpi=160)
            ax.plot(np.linspace(1, duration, duration), net_load, linewidth=1.5, label='Customer load')
            ax.plot(np.linspace(1, duration, duration), net_load-net_pv, linewidth=1.5, label='Customer load+PV')
            ax.plot(np.linspace(1, duration, duration), P_customer, linewidth=1.5, label='Customer load+PV+BESS')
            ax2 = ax.twinx()
            ax2.plot(np.linspace(1, duration, duration), self.price_forecast, linewidth=1.5, label='Price')
            ax.set_xlabel('Time', size=12)
            ax.set_ylabel('Customer Demand (W)', size=12)
            ax2.set_ylabel('TOU price ($/kW)', size=12)
            ax.legend(loc='upper left')
            plt.grid(True)
            plt.show()
            fig.tight_layout()

            fig, ax = plt.subplots(len(self.BESS), 1, figsize=(16, 10), dpi=160)
            for i in range(len(self.BESS)):
                ax.set_title('Battery:' + str(i) + '--Status')
                ax.plot(np.linspace(1, duration, duration), Battery_power[i, :], 'r-', linewidth=1.5, label='BESS Input/Output')
                ax.set_xlabel('Time', size=12)
                ax.set_ylabel('Battery Power', size=12)
                ax2 = ax.twinx()
                ax2.plot(np.linspace(1, duration, duration), Battery_SOC[i, 1:], 'g--', linewidth=1.5, label='BESS SOC')
                ax2.set_xlabel('Time', size=12)
                ax2.set_ylabel('Battery SOC', size=12)
                plt.grid(True)
            plt.show()
            fig.tight_layout()

    def write_BESS_schedules(self, directory='outputs'):

        for i, batt in enumerate(self.BESS):
            output_dir = os.path.dirname(os.getcwd()) + '/' + directory + '/'
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)

            np.savetxt(output_dir+batt+"_schedule_.csv", self.BESS[batt]['schedule'], delimiter=",")

        logger.info("HEMS Agent {}: Saving schedules for dispatch".format(self.name))

    def run(self, action, **kwargs):
        res = {'success': False}
        if f'{action}' in HEMS.__dict__:
            res['result'] = HEMS.__dict__[f'{action}'](self, **kwargs)
            res['success'] = True
        return



HEMS1 = HEMS('Agnet1', 's100c_11', optimization=True)
HEMS1.run(action="optimize_battery", interval=1, duration=24, plot_schedules=False)
HEMS1.run(action="write_BESS_schedules", directory='outputs')
HEMS1.run(action="adjust_battery_dispatch")
#HEMS('A', 's9a_9')