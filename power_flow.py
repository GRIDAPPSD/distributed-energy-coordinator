# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:11:21 2021

@author: poud579
"""

import numpy as np
import cvxpy as cp
import math
# from tabulate import tabulate
# import mosek


class AreaAlpha(object):
    """
    asdf
    """

    def __init__(self):
        self.name = 'area'

    # Optimization to regulate the voltage
    def area_alpha(self, branch_sw_data_case, bus_info, agent_bus, agent_bus_idx, vsrc, bus_info_area_1, bus_info_area_2,
                 bus_info_area_3, bus_info_area_4, bus_info_area_5, pf_flag):

        # Forming the optimization variables
        nbranch = len(branch_sw_data_case)
        nbus = len(bus_info)

        # Find the ABC phase and s1s2 phase triplex line and bus numbers
        nbranch_ABC = 0
        nbus_ABC = 0
        nbranch_s1s1 = 0
        nbus_s1s2 = 0
        secondary_model = ['TPX_LINE', 'SPLIT_PHASE']
        name = []
        for b in branch_sw_data_case:
            if branch_sw_data_case[b]['type'] in secondary_model:
                nbranch_s1s1 += 1
            else:
                nbranch_ABC += 1

        for b in bus_info:
            name.append(b)
            if bus_info[b]['kv'] > 0.4:
                nbus_ABC += 1
            else:
                nbus_s1s2 += 1

        # Number of decision variables
        #             Voltage                     PQ_inj                              PQ_flow
        n = (nbus_ABC * 3 + nbus_s1s2) + (nbus_ABC * 6 + nbus_s1s2 * 2) + (nbranch_ABC * 6 + nbranch_s1s1 * 2) + \
            (nbus_ABC + nbus_s1s2)
        #         alpha_PV

        # print(nbus_ABC, nbus_s1s2, nbranch_ABC, nbranch_s1s1)
        x = cp.Variable(n)

        # Number of equality/inequality constraints (Injection equations (ABC) at each bus)
        p = n + (nbus_ABC * 3 + nbus_s1s2) + (nbus_ABC * 6 + nbus_s1s2 * 2) + (nbus_ABC + nbus_s1s2)
        p = 8000
        # Initialize the matrices
        P = np.zeros((n, n))
        q = np.zeros(n)
        A = np.zeros((p, n))
        b = np.zeros(p)
        G = np.zeros((4000, n))
        h = np.zeros(4000)
        n_alpha = (nbus_ABC * 3 + nbus_s1s2) + (nbus_ABC * 6 + nbus_s1s2 * 2) + (nbranch_ABC * 6 + nbranch_s1s1 * 2)

        for k in range(nbus_ABC + nbus_s1s2):
            q[n_alpha + k] = 1
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

        # Define BFM constraints for both real and reactive power
        # print("Formulating power flow constraints")
        counteq = 0
        baseS = 1 / (1000000 * 100 / 3)
        for keyb, val_bus in bus_info.items():
            if keyb != agent_bus:
                k_frm_3p = []
                k_to_3p = []
                k_frm_1p = []
                k_frm_1pa, k_frm_1pb, k_frm_1pc = [], [], []
                k_frm_1qa, k_frm_1qb, k_frm_1qc = [], [], []
                k_to_1p = []
                # Find bus idx in "from" of branch_sw_data
                ind_frm = 0
                ind_to = 0
                if val_bus['kv'] < 0.4:
                    for key, val_br in branch_sw_data_case.items():
                        if val_bus['idx'] == val_br['from']:
                            k_frm_1p.append(ind_frm - nbranch_ABC)

                        if val_bus['idx'] == val_br['to']:
                            k_to_1p.append(ind_to - nbranch_ABC)
                        ind_to += 1
                        ind_frm += 1
                    loc = (nbus_ABC * 3 + nbus_s1s2) + (nbus_ABC * 6 + nbus_s1s2 * 2) + nbranch_ABC * 6
                    A, b = power_balance(A, b, k_frm_1p, k_to_1p, counteq, loc,
                                              val_bus['idx'] + nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 6)
                    counteq += 1
                    A, b = reac_power_balance(A, b, k_frm_1p, k_to_1p, counteq, loc + nbranch_s1s1,
                                              val_bus['pq'][1] * baseS)
                    counteq += 1
                else:
                    for key, val_br in branch_sw_data_case.items():
                        if val_bus['idx'] == val_br['from']:
                            if bus_info[val_br['to_bus']]['kv'] > 0.4:
                                k_frm_3p.append(ind_frm)
                            else:
                                if key[-1] == 'a':
                                    k_frm_1pa.append(nbranch_ABC * 6 + ind_frm - nbranch_ABC)
                                    k_frm_1qa.append(nbranch_ABC * 3 + ind_frm - nbranch_ABC + nbranch_s1s1)
                                if key[-1] == 'b':
                                    k_frm_1pb.append(nbranch_ABC * 5 + ind_frm - nbranch_ABC)
                                    k_frm_1qb.append(nbranch_ABC * 2 + ind_frm - nbranch_ABC + nbranch_s1s1)
                                if key[-1] == 'c':
                                    k_frm_1pc.append(nbranch_ABC * 4 + ind_frm - nbranch_ABC)
                                    k_frm_1qc.append(nbranch_ABC * 1 + ind_frm - nbranch_ABC + nbranch_s1s1)

                        if val_bus['idx'] == val_br['to']:
                            if bus_info[val_br['fr_bus']]['kv'] > 0.4:
                                k_to_3p.append(ind_to)
                            else:
                                k_to_1p.append(ind_to - nbranch_ABC)
                        ind_to += 1
                        ind_frm += 1
                    loc = (nbus_ABC * 3 + nbus_s1s2) + (nbus_ABC * 6 + nbus_s1s2 * 2)
                    # Finding the kfrms
                    k_frm_A = k_frm_3p + k_frm_1pa
                    k_frm_B = k_frm_3p + k_frm_1pb
                    k_frm_C = k_frm_3p + k_frm_1pc
                    k_to_A = k_to_B = k_to_C = k_to_3p
                    # Real Power balance equations
                    # Phase A
                    A, b = power_balance(A, b, k_frm_A, k_to_A, counteq, loc,
                                         val_bus['idx'] + nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 0)
                    counteq += 1
                    # # Phase B
                    A, b = power_balance(A, b, k_frm_B, k_to_B, counteq, loc + nbranch_ABC,
                                         val_bus['idx'] + nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 1)
                    counteq += 1
                    # # Phase C
                    A, b = power_balance(A, b, k_frm_C, k_to_C, counteq, loc + nbranch_ABC * 2,
                                         val_bus['idx'] + nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 2)
                    counteq += 1
                    k_frm_A = k_frm_3p + k_frm_1qa
                    k_frm_B = k_frm_3p + k_frm_1qb
                    k_frm_C = k_frm_3p + k_frm_1qc
                    # Reactive Power balance equations
                    loc = (nbus_ABC * 3 + nbus_s1s2) + (nbus_ABC * 6 + nbus_s1s2 * 2) + nbranch_ABC * 3
                    # Phase A
                    A, b = reac_power_balance(A, b, k_frm_A, k_to_A, counteq, loc,
                                              val_bus['pq'][0][1] * baseS)
                    counteq += 1
                    # Phase B
                    A, b = reac_power_balance(A, b, k_frm_B, k_to_B, counteq, loc + nbranch_ABC,
                                              val_bus['pq'][1][1] * baseS)
                    counteq += 1
                    # Phase C
                    A, b = reac_power_balance(A, b, k_frm_C, k_to_C, counteq, loc + nbranch_ABC * 2,
                                              val_bus['pq'][2][1] * baseS)
                    counteq += 1

        # Make injection a decision variable
        # print("Formulating injection constraints")
        for keyb, val_bus in bus_info.items():
            if keyb != agent_bus:
                # Real power injection at a bus
                if val_bus['kv'] > 0.4:
                    # Phase A
                    A[counteq, nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 0 + val_bus['idx']] = 1
                    A[counteq, n_alpha + val_bus['idx']] = - val_bus['pv'][0][0] * baseS
                    b[counteq] = - val_bus['pv'][0][0] * baseS + val_bus['pq'][0][0] * baseS
                    counteq += 1
                    # Phase B
                    A[counteq, nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 1 + val_bus['idx']] = 1
                    A[counteq, n_alpha + val_bus['idx']] = - val_bus['pv'][1][0] * baseS
                    b[counteq] = - val_bus['pv'][1][0] * baseS + val_bus['pq'][1][0] * baseS
                    counteq += 1
                    # Phase C
                    A[counteq, nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 2 + val_bus['idx']] = 1
                    A[counteq, n_alpha + val_bus['idx']] = - val_bus['pv'][2][0] * baseS
                    b[counteq] = - val_bus['pv'][2][0] * baseS + val_bus['pq'][2][0] * baseS
                    counteq += 1
                else:
                    A[counteq, nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 6 + val_bus['idx']] = 1
                    A[counteq, n_alpha + val_bus['idx']] = - val_bus['pv'][0] * baseS
                    b[counteq] = - val_bus['pv'][0] * baseS + val_bus['pq'][0] * baseS
                    counteq += 1

        # Imposing area wise equality constraints
        # AREA 1
        idx = []
        for keyb in bus_info_area_1:
            idx.append(bus_info[keyb]['idx'])
        for k in range(len(idx)-1):
            A[counteq, n_alpha + idx[k]] = 1
            A[counteq, n_alpha + idx[k+1]] = -1
            b[counteq] = 0.0
            counteq += 1
        # AREA 2
        idx = []
        for keyb in bus_info_area_2:
            idx.append(bus_info[keyb]['idx'])
        for k in range(len(idx) - 1):
            A[counteq, n_alpha + idx[k]] = 1
            A[counteq, n_alpha + idx[k + 1]] = -1
            b[counteq] = 0.0
            counteq += 1
        # AREA 3
        idx = []
        for keyb in bus_info_area_3:
            idx.append(bus_info[keyb]['idx'])
        for k in range(len(idx) - 1):
            A[counteq, n_alpha + idx[k]] = 1
            A[counteq, n_alpha + idx[k + 1]] = -1
            b[counteq] = 0.0
            counteq += 1
        # AREA 4
        idx = []
        for keyb in bus_info_area_4:
            idx.append(bus_info[keyb]['idx'])
        for k in range(len(idx) - 1):
            A[counteq, n_alpha + idx[k]] = 1
            A[counteq, n_alpha + idx[k + 1]] = -1
            b[counteq] = 0.0
            counteq += 1
        # AREA 5
        idx = []
        for keyb in bus_info_area_5:
            idx.append(bus_info[keyb]['idx'])
        for k in range(len(idx) - 1):
            A[counteq, n_alpha + idx[k]] = 1
            A[counteq, n_alpha + idx[k + 1]] = -1
            b[counteq] = 0.0
            counteq += 1

        # for k in range(nbus_ABC + nbus_s1s2 - 1):
        #     A[counteq, n_alpha + k] = 1
        #     A[counteq, n_alpha + k + 1] = -1
        #     b[counteq] = 0.0
        #     counteq += 1

        countineq = 0
        for k in range(nbus_ABC + nbus_s1s2):
            G[countineq, n_alpha + k] = 1
            h[countineq] = 1.0
            countineq += 1
        for k in range(nbus_ABC + nbus_s1s2):
            G[countineq, n_alpha + k] = -1
            h[countineq] = -0.0
            countineq += 1

        # Constraint 2: Vj = Vi - Zij Sij* - Sij Zij*
        basez = 4.16 ** 2 / 100

        def voltage_cons_pri(A, b, p, frm, to, counteq, pii, qii, pij, qij, pik, qik):
            A[counteq, frm] = 1
            A[counteq, to] = -1
            n_flow_ABC = (nbus_ABC * 3 + nbus_s1s2) + (nbus_ABC * 6 + nbus_s1s2 * 2)
            # real power drop
            A[counteq, p + n_flow_ABC] = pii / basez
            A[counteq, p + n_flow_ABC + nbranch_ABC] = pij / basez
            A[counteq, p + n_flow_ABC + nbranch_ABC * 2] = pik / basez
            # reactive power drop
            A[counteq, p + n_flow_ABC + nbranch_ABC * 3] = qii / basez
            A[counteq, p + n_flow_ABC + nbranch_ABC * 4] = qij / basez
            A[counteq, p + n_flow_ABC + nbranch_ABC * 5] = qik / basez
            b[counteq] = 0.0
            return A, b

        # Write the voltage constraints for connected branches only
        # print("Formulating voltage constraints")
        idx = 0
        v_lim = []
        for k, val_br in branch_sw_data_case.items():
            # Not writing voltage constraints for transformers
            if val_br['type'] not in secondary_model:
                z = np.asarray(val_br['zprim'])
                v_lim.append(val_br['from'])
                v_lim.append(val_br['to'])
                # Writing three phase voltage constraints
                # Phase A
                paa, qaa = -2 * z[0, 0][0], -2 * z[0, 0][1]
                pab, qab = -(- z[0, 1][0] + math.sqrt(3) * z[0, 1][1]), -(
                        - z[0, 1][1] - math.sqrt(3) * z[0, 1][0])
                pac, qac = -(- z[0, 2][0] - math.sqrt(3) * z[0, 2][1]), -(
                        - z[0, 2][1] + math.sqrt(3) * z[0, 2][0])
                A, b = voltage_cons_pri(A, b, idx, val_br['from'], val_br['to'], counteq, paa, qaa, pab, qab, pac, qac)
                counteq += 1
                # Phase B
                pbb, qbb = -2 * z[1, 1][0], -2 * z[1, 1][1]
                pba, qba = -(- z[0, 1][0] - math.sqrt(3) * z[0, 1][1]), -(
                        - z[0, 1][1] + math.sqrt(3) * z[0, 1][0])
                pbc, qbc = -(- z[1, 2][0] + math.sqrt(3) * z[1, 2][1]), -(
                        - z[1, 2][1] - math.sqrt(3) * z[1, 2][0])
                A, b = voltage_cons_pri(A, b, idx, nbus_ABC + val_br['from'], nbus_ABC + val_br['to'], counteq, pba,
                                        qba,
                                        pbb, qbb, pbc, qbc)
                counteq += 1
                # Phase C
                pcc, qcc = -2 * z[2, 2][0], -2 * z[2, 2][1]
                pca, qca = -(- z[0, 2][0] + math.sqrt(3) * z[0, 2][1]), -(
                        - z[0, 2][1] - math.sqrt(3) * z[0, 2][0])
                pcb, qcb = -(- z[1, 2][0] - math.sqrt(3) * z[1, 2][1]), -(
                        - z[0, 2][1] + math.sqrt(3) * z[1, 2][0])
                A, b = voltage_cons_pri(A, b, idx, nbus_ABC * 2 + val_br['from'], nbus_ABC * 2 + val_br['to'], counteq,
                                        pca, qca, pcb, qcb, pcc, qcc)
                counteq += 1
            idx += 1

        def voltage_cons(A, b, p, frm, to, counteq, p_pri, q_pri, p_sec, q_sec):
            A[counteq, frm] = 1
            A[counteq, to] = -1
            n_flow_s1s2 = (nbus_ABC * 3 + nbus_s1s2) + (nbus_ABC * 6 + nbus_s1s2 * 2) + nbranch_ABC * 6
            # real power drop
            A[counteq, p + n_flow_s1s2] = p_pri + 0.5 * p_sec
            # reactive power drop
            A[counteq, p + n_flow_s1s2 + nbranch_s1s1] = q_pri + 0.5 * q_sec
            b[counteq] = 0.0
            return A, b

        # Write the constraints for connected branches only
        idx = 0
        # v_lim = []
        pq_index = []
        for k, val_br in branch_sw_data_case.items():
            # For split phase transformer, we use interlace design
            if val_br['type'] in secondary_model:
                if val_br['type'] == 'SPLIT_PHASE':
                    pq_index.append(val_br['idx'])
                    zp = np.asarray(val_br['impedance'])
                    zs = np.asarray(val_br['impedance1'])
                    v_lim.append(val_br['from'])
                    v_lim.append(val_br['to'])
                    # Writing voltage constraints
                    # Phase S1
                    p_pri, q_pri = -2 * zp[0], -2 * zp[1]
                    p_sec, q_sec = -2 * zs[0], -2 * zs[1]
                    phase = k[-1]
                    if phase == 'a':
                        from_bus = val_br['from']
                    if phase == 'b':
                        from_bus = val_br['from'] + nbus_ABC
                    if phase == 'c':
                        from_bus = val_br['from'] + nbus_ABC * 2
                    to_bus = val_br['to'] - nbus_ABC + nbus_ABC * 3
                    A, b = voltage_cons(A, b, idx - nbus_ABC, from_bus, to_bus, counteq, p_pri, q_pri, p_sec, q_sec)
                    counteq += 1

                # For triplex line, we assume there is no mutual coupling
                if val_br['type'] != 'SPLIT_PHASE':
                    # The impedance of line will be converted into pu here.
                    zbase = 120.0 * 120.0 / 15000
                    zp = np.asarray(val_br['impedance'])
                    v_lim.append(val_br['from'])
                    v_lim.append(val_br['to'])
                    # Writing voltage constraints
                    # Phase S1
                    p_s1, q_s1 = 0, 0
                    p_s2, q_s2 = -2 * zp[0, 0][0] / zbase, -2 * zp[0, 0][1] / zbase
                    from_bus = val_br['from'] - nbus_ABC + nbus_ABC * 3
                    to_bus = val_br['to'] - nbus_ABC + nbus_ABC * 3
                    A, b = voltage_cons(A, b, idx - nbus_ABC, from_bus, to_bus, counteq, p_s1, q_s1, p_s2, q_s2)
                    counteq += 1
            idx += 1

        # The node after substation transformer is where we start the voltage constraints
        A[counteq, agent_bus_idx] = 1
        b[counteq] = vsrc[0] ** 2
        counteq += 1
        A[counteq, agent_bus_idx + nbus_ABC] = 1
        b[counteq] = vsrc[1] ** 2
        counteq += 1
        A[counteq, agent_bus_idx + nbus_ABC * 2] = 1
        b[counteq] = vsrc[2] ** 2
        counteq += 1

        # Constraint 3: 0.95^2 <= V <= 1.05^2 (For those nodes where voltage constraint exist)
        # print("Formulating voltage limit constraints")
        v_idxs = list(set(v_lim))
        # # TODO: Does the vmin make sense here?
        if pf_flag == 1:
            vmax = 1.5
        else:
            vmax = 1.05
        vmin = 0.95
        for k in range(nbus_ABC):
            if k in v_idxs:
                # Upper bound
                G[countineq, k] = 1
                h[countineq] = vmax ** 2
                countineq += 1
                G[countineq, k + nbus_ABC] = 1
                h[countineq] = vmax ** 2
                countineq += 1
                G[countineq, k + nbus_ABC * 2] = 1
                h[countineq] = vmax ** 2
                countineq += 1
                # Lower Bound
                G[countineq, k] = -1
                h[countineq] = - vmin ** 2
                countineq += 1
                G[countineq, k + nbus_ABC] = -1
                h[countineq] = - vmin ** 2
                countineq += 1
                G[countineq, k + nbus_ABC * 2] = -1
                h[countineq] = - vmin ** 2
                countineq += 1

        prob = cp.Problem(cp.Minimize(q.T @ x),
                          [G @ x <= h,
                           A @ x == b])

        # prob.solve(solver=cp.ECOS, verbose=False, max_iters=500, abstol=1e-4, reltol=1e-4, feastol=1e-4)
        prob.solve(solver=cp.MOSEK, verbose=False)
        # print(prob.status)
        if prob.status == 'infeasible':
            print("Check for limits. Power flow didn't converge")
            return 0

        # Printing the line flows
        from_bus = []
        to_bus = []
        name = []
        for key, val_br in branch_sw_data_case.items():
            from_bus.append(val_br['fr_bus'])
            to_bus.append(val_br['to_bus'])
            name.append(key)
        # print('\n Real and Reactive Power flow:')
        i = 0
        flow = []
        mul = 1 / (baseS * 1000)
        line_flow = {}
        n_flow_ABC = (nbus_ABC * 3 + nbus_s1s2) + (nbus_ABC * 6 + nbus_s1s2 * 2)
        for k in range(n_flow_ABC, n_flow_ABC + nbranch_ABC):
            flow.append([name[i], from_bus[i], to_bus[i], '{:.3f}'.format(x.value[k] * mul),
                         '{:.3f}'.format(x.value[k + nbranch_ABC] * mul),
                         '{:.3f}'.format(x.value[k + nbranch_ABC * 2] * mul),
                         '{:.3f}'.format(x.value[k + nbranch_ABC * 3] * mul),
                         '{:.3f}'.format(x.value[k + nbranch_ABC * 4] * mul),
                         '{:.3f}'.format(x.value[k + nbranch_ABC * 5] * mul)])
            line_flow[name[i]] = {}
            line_flow[name[i]]['A'] = [x.value[k] * mul * 1000, x.value[k + nbranch_ABC * 3] * mul * 1000]
            line_flow[name[i]]['B'] = [x.value[k + nbranch_ABC] * mul * 1000, x.value[k + nbranch_ABC * 4] * mul * 1000]
            line_flow[name[i]]['C'] = [x.value[k + nbranch_ABC * 2] * mul * 1000,
                                       x.value[k + nbranch_ABC * 5] * mul * 1000]
            i += 1
        # print(tabulate(flow, headers=['Line Name', 'from', 'to', 'P_A', 'P_B', 'P_C', 'Q_A', 'Q_B', 'Q_C'],
        #                tablefmt='psql'))
        n_flow_s1s2 = (nbus_ABC * 3 + nbus_s1s2) + (nbus_ABC * 6 + nbus_s1s2 * 2) + nbranch_ABC * 6
        flow = []
        for k in range(n_flow_s1s2, n_flow_s1s2 + nbranch_s1s1):
            flow.append([name[i], from_bus[i], to_bus[i], '{:.3f}'.format(x.value[k] * mul),
                         '{:.3f}'.format(x.value[k + nbranch_s1s1] * mul)])
            i += 1
        # print(tabulate(flow, headers=['Line Name', 'from', 'to', 'P_S1S2', 'Q_S1S2'], tablefmt='psql'))

        # print('\n Voltages at buses:')
        name = []
        for key, val_br in bus_info.items():
            name.append(key)
        volt = []
        bus_voltage = {}
        i = 0
        for k in range(nbus_ABC):
            volt.append(
                [name[k], '{:.4f}'.format(math.sqrt(abs(x.value[k]))),
                 '{:.4f}'.format(math.sqrt(abs(x.value[nbus_ABC + k]))),
                 '{:.4f}'.format(math.sqrt(abs(x.value[nbus_ABC * 2 + k])))])
            bus_voltage[name[k]] = {}
            bus_voltage[name[k]]['A'] = math.sqrt(abs(x.value[k]))
            bus_voltage[name[k]]['B'] = math.sqrt(abs(x.value[nbus_ABC + k]))
            bus_voltage[name[k]]['C'] = math.sqrt(abs(x.value[nbus_ABC * 2 + k]))
            i += 1
        # print(tabulate(volt, headers=['Bus Name', 'V_A', 'V_B', 'V_C', 'V_A (pu)', 'V_B (pu)', 'V_C (pu)'],
        #                tablefmt='psql'))

        n_volt_s1s2 = nbus_ABC * 3
        volt = []
        for k in range(nbus_s1s2):
            volt.append(
                [name[i], '{:.7f}'.format(math.sqrt(abs(x.value[k + n_volt_s1s2])))])
            i += 1
        # print(tabulate(volt, headers=['Bus Name', 'V_S'],
        #                 tablefmt='psql'))

        # print('\n Injections at buses:')
        injection = []
        alpha = []
        for k in range(nbus_ABC + nbus_s1s2):
            injection.append([name[k], '{:.4f}'.format((x.value[k + n_alpha]))])
            alpha.append(x.value[k + n_alpha])

        # print(tabulate(injection, headers=['Bus Name', 'Alpha_PV'], tablefmt='psql'))
        return bus_voltage, [alpha[12], alpha[52], alpha[64], alpha[81], alpha[117]]
