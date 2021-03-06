# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:11:21 2021

@author: poud579
"""

import numpy as np
import cvxpy as cp
import math
from tabulate import tabulate
from HEMS_agent import HEMS

class Secondary_Agent(object):
    """
    asdf
    """
    def __init__(self, sourcebus, sp_graph, bus_info_sec, tpx_xfmr, phase, feeder_mrid, load, pv, batt):

        self.name = sourcebus
        self.phase = phase
        for k in sp_graph:
            if sourcebus in k:
                area = k
                break
        self.bus_info_sec_agent_i = {}
        idx = 0
        for key, val_bus in bus_info_sec.items():
            if key in area:
                self.bus_info_sec_agent_i[key] = {}
                self.bus_info_sec_agent_i[key]['idx'] = idx
                self.bus_info_sec_agent_i[key]['phase'] = bus_info_sec[key]['phases']
                self.bus_info_sec_agent_i[key]['nodes'] = bus_info_sec[key]['nodes']
                self.bus_info_sec_agent_i[key]['injection'] = bus_info_sec[key]['injection']
                self.bus_info_sec_agent_i[key]['pv'] = bus_info_sec[key]['pv']
                self.bus_info_sec_agent_i[key]['pq'] = bus_info_sec[key]['pq']
                idx += 1

        idx = 0
        self.tpx_xfmr_agent_i = {}
        for key, val_bus in tpx_xfmr.items():
            if val_bus['fr_bus'] in self.bus_info_sec_agent_i and val_bus['to_bus'] in self.bus_info_sec_agent_i:
                self.tpx_xfmr_agent_i[key] = {}
                self.tpx_xfmr_agent_i[key]['idx'] = idx
                self.tpx_xfmr_agent_i[key]['type'] = tpx_xfmr[key]['type']
                self.tpx_xfmr_agent_i[key]['from'] = self.bus_info_sec_agent_i[tpx_xfmr[key]['fr_bus']]['idx']
                self.tpx_xfmr_agent_i[key]['to'] = self.bus_info_sec_agent_i[tpx_xfmr[key]['to_bus']]['idx']
                self.tpx_xfmr_agent_i[key]['fr_bus'] = tpx_xfmr[key]['fr_bus']
                self.tpx_xfmr_agent_i[key]['to_bus'] = tpx_xfmr[key]['to_bus']
                if tpx_xfmr[key]['type'] == 'SPLIT_PHASE':
                    self.xfmr_name = key
                    self. tpx_xfmr_agent_i[key]['impedance'] = tpx_xfmr[key]['impedance']
                    self.tpx_xfmr_agent_i[key]['impedance1'] = tpx_xfmr[key]['impedance1']
                else:
                    self.tpx_xfmr_agent_i[key]['impedance'] = tpx_xfmr[key]['impedance']
                # branch_sw_data_sec_agent_i[key]['zprim'] = branch_sw_data[key]['zprim']
                idx += 1


        self.HEMS_list = {}
        for key, val_bus in self.bus_info_sec_agent_i.items():
            self.HEMS_list[key] = HEMS('HEMS'+key, key.lower(), feeder_mrid, load, pv, batt)


    # Optimization to regulate the voltage
    def alpha_area(self, agent_bus, agent_bus_idx, vsrc, service_xfmr_bus, xfmr_phase, ratedS):
        xfmr_tpx = self.tpx_xfmr_agent_i
        bus_info = self.bus_info_sec_agent_i

        # Forming the optimization variables
        nbranch = len(xfmr_tpx)
        nbus = len(bus_info)

        # Number of decision variables
        # Voltage     PQ Inj      PQ Flow   PV_cur alpha
        n = nbus + nbus * 2 + nbranch * 2 + nbus + nbus

        # Number of equality/inequality constraints
        p = nbranch * 25 + nbus * 6

        # Initialize the matrices
        P = np.random.randn(n, n) * 0
        q = np.random.randn(n) * 0
        A = np.zeros((p, n))
        b = np.zeros(p)
        G = np.zeros((p, n))
        h = np.zeros(p)

        # Formulate the objective function
        q[nbus * 1 + nbus * 2 + nbranch * 2 + nbus] = 1
        P[nbus * 1 + nbus * 2 + nbranch * 2 + nbus, nbus * 1 + nbus * 2 + nbranch * 2 + nbus] = 0

        # Define the constraints
        # Constraint 1: sum(Sij) - sum(Sjk) == -sj
        def power_balance(A, b, k_frm, k_to, counteq, col, val):
            for k in k_frm:
                A[counteq, col + k] = -1
            for k in k_to:
                A[counteq, col + k] = 1

            A[counteq, val] = -1
            b[counteq] = 0
            return A, b

        def reac_power_balance(A, b, k_frm, k_to, counteq, col, val):
            for k in k_frm:
                A[counteq, col + k] = -1
            for k in k_to:
                A[counteq, col + k] = 1

            b[counteq] = val
            return A, b

        # Define it for both real and reactive power
        counteq = 0
        baseS = 1 / ratedS
        for keyb, val_bus in bus_info.items():
            if agent_bus != keyb:
                k_frm = []
                k_to = []
                # Find bus idx in "from" of branch_sw_data
                ind_frm = 0
                ind_to = 0
                for key, val_br in xfmr_tpx.items():
                    if val_bus['idx'] == val_br['from']:
                        k_frm.append(ind_frm)
                    if val_bus['idx'] == val_br['to']:
                        k_to.append(ind_to)
                    ind_to += 1
                    ind_frm += 1
                # Real Power balance equations
                # TODO: Include the PV contribution here. PV power will be a decision variable afterwards
                # Phase S1
                A, b = power_balance(A, b, k_frm, k_to, counteq, nbus * 3, val_bus['idx'] + nbus * 1)
                counteq += 1

                # Reactive Power balance equations
                # Phase S1
                # A, b = power_balance(A, b, k_frm, k_to, counteq, nbus*3+nbranch, val_bus['idx'] + nbus*2)
                A, b = reac_power_balance(A, b, k_frm, k_to, counteq, nbus * 3 + nbranch, val_bus['pq'].imag * baseS)
                counteq += 1

        # Make injection a decision variable
        countineq = 0
        for keyb, val_bus in bus_info.items():
            if agent_bus != keyb:
                # Net PV power at a node (PV_net = P_PV - alpha * P_PV)
                A[counteq, nbus * 1 + nbus * 2 + nbranch * 2 + val_bus['idx']] = 1
                A[counteq, nbus * 1 + nbus * 2 + nbranch * 2 + nbus + val_bus['idx']] = val_bus['pv'].real * baseS
                b[counteq] = val_bus['pv'].real * baseS
                counteq += 1

                # Real power injection at a bus (P_inj = P_load - PV_net)
                A[counteq, nbus * 1 + val_bus['idx']] = 1
                A[counteq, nbus * 1 + nbus * 2 + nbranch * 2 + val_bus['idx']] = 1
                b[counteq] = val_bus['pq'].real * baseS
                counteq += 1

        # Enforce all alpha to be equal by a secondary agent. Might not be feasible always.
        # Allow different alpha if voltage is very high
        if vsrc[0] < 1.06:
            for k in range(nbus - 1):
                A[counteq, nbus * 1 + nbus * 2 + nbranch * 2 + nbus + k] = 1
                A[counteq, nbus * 1 + nbus * 2 + nbranch * 2 + nbus + k + 1] = -1
                b[counteq] = 0
                counteq += 1

        # Alpha cannot be negative
        for k in range(nbus):
            G[countineq, nbus * 1 + nbus * 2 + nbranch * 2 + nbus + k] = -1
            h[countineq] = 0
            countineq += 1
            G[countineq, nbus * 1 + nbus * 2 + nbranch * 2 + nbus + k] = 1
            h[countineq] = 1
            countineq += 1

        # Constraint 2: Vj = Vi - Zij Sij* - Sij Zij*
        def voltage_cons(A, b, p, frm, to, counteq, p_pri, q_pri, p_sec, q_sec):
            A[counteq, frm] = 1
            A[counteq, to] = -1
            # real power drop
            A[counteq, p + nbus * 3] = p_pri + 0.5 * p_sec
            # reactive power drop
            A[counteq, p + nbus * 3 + nbranch] = q_pri + 0.5 * q_sec
            b[counteq] = 0.0
            return A, b

        # Write the constraints for connected branches only
        idx = 0
        v_lim = []
        pq_index = []
        for k, val_br in xfmr_tpx.items():
            # For split phase transformer, we use interlace design
            if val_br['type'] == 'SPLIT_PHASE':
                pq_index.append(val_br['idx'])
                zp = val_br['impedance']
                zs = val_br['impedance1']
                v_lim.append(val_br['from'])
                v_lim.append(val_br['to'])
                # Writing voltage constraints
                # Phase S1
                p_pri, q_pri = -2 * zp[0].real, -2 * zp[0].imag
                p_sec, q_sec = -2 * zs[0].real, -2 * zs[0].imag
                A, b = voltage_cons(A, b, idx, val_br['from'], val_br['to'], counteq, p_pri, q_pri, p_sec, q_sec)
                counteq += 1

            # For triplex line, we assume there is no mutual coupling
            if val_br['type'] != 'SPLIT_PHASE':
                # The impedance of line will be converted into pu here. 
                zbase = 120.0 * 120.0 / ratedS
                zp = val_br['impedance']
                v_lim.append(val_br['from'])
                v_lim.append(val_br['to'])
                # Writing voltage constraints
                # Phase S1
                p_s1, q_s1 = 0, 0
                p_s2, q_s2 = -2 * zp[0].real / zbase, -2 * zp[0].imag / zbase
                A, b = voltage_cons(A, b, idx, val_br['from'], val_br['to'], counteq, p_s1, q_s1, p_s2, q_s2)
                counteq += 1
            idx += 1

        # The node where service transformer is where we start the voltage constraints
        A[counteq, agent_bus_idx] = 1
        b[counteq] = vsrc[0] ** 2
        counteq += 1

        # Constraint 3: 0.95^2 <= V <= 1.05^2 (For those nodes where voltage constraint exist)
        v_idxs = list(set(v_lim))
        # print(v_idxs)
        for k in range(nbus):
            if k in v_idxs and k != agent_bus_idx:
                # Upper bound
                G[countineq, k] = 1
                h[countineq] = 1.05 ** 2
                countineq += 1
                # Lower Bound
                G[countineq, k] = -1
                h[countineq] = -0.95 ** 2
                countineq += 1

        x = cp.Variable(n)
        # prob = cp.Problem(cp.Minimize(1 * cp.quad_form(x, P) + q.T @ x),
        #                   [G @ x <= h,
        #                    A @ x == b])
        prob = cp.Problem(cp.Minimize(q.T @ x),
                          [G @ x <= h,
                           A @ x == b])

        prob.solve(solver=cp.ECOS, verbose=False, max_iters=500, feastol=1e-4)
        # print("\nThe optimal value is", (prob.value), prob.status)

        # Printing the line flows
        from_bus = []
        to_bus = []
        name = []
        for key, val_br in xfmr_tpx.items():
            from_bus.append(val_br['fr_bus'])
            to_bus.append(val_br['to_bus'])
            name.append(key)
        # print('\n Real and Reactive Power flow:')
        i = 0
        flow = []
        mul = 1 / (baseS * 1000)
        for k in range(nbus * 3, nbus * 3 + nbranch):
            flow.append([name[i], from_bus[i], to_bus[i], '{:.3f}'.format(x.value[k] * mul),
                         '{:.3f}'.format(x.value[k + nbranch] * mul)])
            i += 1
        # print(tabulate(flow, headers=['Line Name', 'from', 'to', 'P_s1s2', 'Q_s1s2'], tablefmt='psql'))

        # Extract the equivalent injections from service transformer bus
        if len(pq_index) < 2:
            pq_inj = x.value[nbus * 3 + pq_index[0]] * mul * 1000 + 1j * x.value[
                nbus * 3 + nbranch + pq_index[0]] * mul * 1000
            if xfmr_phase == 'A':
                sec_inj = [pq_inj, complex(0, 0), complex(0, 0)]
            elif xfmr_phase == 'B':
                sec_inj = [complex(0, 0), pq_inj, complex(0, 0)]
            else:
                sec_inj = [complex(0, 0), complex(0, 0), pq_inj]
        else:
            # if len(pq_index) is > 2, we hardcode for a bus which has three split phase transformers connected to it.
            print(service_xfmr_bus[agent_bus]['phase'])
            print(pq_index)
            pq_inj_A = x.value[nbus * 3 + pq_index[0]] * mul * 1000 + 1j * x.value[
                nbus * 3 + nbranch + pq_index[0]] * mul * 1000
            pq_inj_B = x.value[nbus * 3 + pq_index[1]] * mul * 1000 + 1j * x.value[
                nbus * 3 + nbranch + pq_index[1]] * mul * 1000
            pq_inj_C = x.value[nbus * 3 + pq_index[2]] * mul * 1000 + 1j * x.value[
                nbus * 3 + nbranch + pq_index[2]] * mul * 1000
            sec_inj = [pq_inj_A, pq_inj_B, pq_inj_C]

        # Print node voltages
        name = []
        for key, val_br in bus_info.items():
            name.append(key)
        # print('\n Voltages at buses:')
        volt = []
        bus_voltage = {}
        try:
            for k in range(nbus):
                volt.append([name[k], '{:.4f}'.format(math.sqrt(x.value[k]))])
                bus_voltage[name[k]] = {}
                bus_voltage[name[k]]['voltage'] = math.sqrt(x.value[k])
            # print(tabulate(volt, headers=['Bus Name', 'V_S1'], tablefmt='psql'))
        except:
            pass
        # print('\n Injections at buses:')
        injection = []
        for k in range(nbus):
            injection.append([name[k], '{:.3f}'.format((x.value[k + nbus]) * mul),
                              '{:.3f}'.format((x.value[k + nbus * 3 + nbranch * 2 + nbus]))])
        # print(tabulate(injection, headers=['Bus Name', 'P_inj', 'Alpha'], tablefmt='psql'))

        # sum = 0.0
        # for i in range(nbranch * 100 + nbus * 6):
        #     s = 0.0
        #     for k in range(nbus * 1  + nbus * 2 + nbranch * 2 + nbus):
        #         s += A[i, k] * x.value[k] 
        #     sum += s - b[i]
        #     #print(s, b[i])  

        # print("\n The Ax-b expression sum is:", sum, "\n")

        objective = (prob.value)
        status = prob.status
        alpha = x.value[nbus * 1 + nbus * 2 + nbranch * 2 + nbus]
        return sec_inj, alpha, bus_voltage
