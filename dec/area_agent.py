# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:11:21 2021

@author: poud579
"""

import numpy as np
import cvxpy as cp
import math
from tabulate import tabulate


class AreaCoordinator(object):
    """
    asdf
    """

    def __init__(self):
        self.name = 'area'

    # Optimization to regulate the voltage
    def alpha_area(self, branch_sw_data_case, bus_info, agent_bus, agent_bus_idx, vsrc, service_xfmr_bus):

        # Forming the optimization variables
        nbranch = len(branch_sw_data_case)
        nbus = len(bus_info)

        # Number of decision variables
        #     V_ABC      PQ_inj_ABC    PQ_flow_ABC       alpha_inj   alpha_PV
        n = nbus * 3 +   nbus * 6   +   nbranch * 6   +     nbus   +   nbus

        # Number of equality/inequality constraints (Injection equations (ABC) at each bus)
        p = nbranch * 25 + nbus * 6
        # Initialize the matrices
        P = np.zeros((n, n))
        q = np.zeros(n)
        A = np.zeros((p, n))
        b = np.zeros(p)
        G = np.zeros((p, n))
        h = np.zeros(p)

        # P = P.T @ P
        # The objective function is written inside P
        # TODO How will the maximize problem look like
        # print("Formulating objective function")
        for keyb, val_bus in bus_info.items():
        #     if keyb not in service_xfmr_bus:
        #         P[nbus * 3 + nbus * 6 + nbranch * 6 + val_bus['idx'], nbus * 3 + nbus * 6 + nbranch * 6 + val_bus['idx']] = 0
        #         q[nbus * 3 + nbus * 6 + nbranch * 6 + val_bus['idx']] = 1
        #     else:
        #         P[nbus * 3 + nbus * 6 + nbranch * 6 + val_bus['idx'], nbus * 3 + nbus * 6 + nbranch * 6 + val_bus['idx']] = 0
        #         q[nbus * 3 + nbus * 6 + nbranch * 6 + val_bus['idx']] = 1
            q[nbus * 3 + nbus * 6 + nbranch * 6 + val_bus['idx']] = 10
            q[nbus * 3 + nbus * 6 + nbranch * 6 + nbus + val_bus['idx']] = 1
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
        # baseS = 1
        for keyb, val_bus in bus_info.items():
            if agent_bus != keyb:
                k_frm = []
                k_to = []
                # Find bus idx in "from" of branch_sw_data
                ind_frm = 0
                ind_to = 0
                for key, val_br in branch_sw_data_case.items():
                    if val_bus['idx'] == val_br['from']:
                        k_frm.append(ind_frm)
                    if val_bus['idx'] == val_br['to']:
                        k_to.append(ind_to)
                    ind_to += 1
                    ind_frm += 1
                # Real Power balance equations
                # Phase A
                A, b = power_balance(A, b, k_frm, k_to, counteq, nbus * 9, val_bus['idx'] + nbus * 3)
                counteq += 1
                # Phase B
                A, b = power_balance(A, b, k_frm, k_to, counteq, nbus * 9 + nbranch, val_bus['idx'] + nbus * 4)
                counteq += 1
                # Phase C
                A, b = power_balance(A, b, k_frm, k_to, counteq, nbus * 9 + nbranch * 2, val_bus['idx'] + nbus * 5)
                counteq += 1

                # Reactive Power balance equations
                # Phase A
                A, b = reac_power_balance(A, b, k_frm, k_to, counteq, nbus * 9 + nbranch * 3,
                                          val_bus['injection'][0].imag * baseS + val_bus['pq'][0].imag * baseS)
                counteq += 1
                # Phase B
                A, b = reac_power_balance(A, b, k_frm, k_to, counteq, nbus * 9 + nbranch * 4,
                                          val_bus['injection'][1].imag * baseS + val_bus['pq'][1].imag * baseS)
                counteq += 1
                # Phase C
                A, b = reac_power_balance(A, b, k_frm, k_to, counteq, nbus * 9 + nbranch * 5,
                                          val_bus['injection'][2].imag * baseS + val_bus['pq'][2].imag * baseS)
                counteq += 1

        # Make injection a decision variable
        # print("Formulating injection constraints")
        countineq = 0
        for keyb, val_bus in bus_info.items():
            if agent_bus != keyb:
                # Real power injection at a bus
                # Net injection at a primary node (Function of injection from XFMR bus and loads & PV in primary network
                # P_inj(net) = Pinj - alpha_1 * Pinj + Pload - (PV - alpha_2 * PV)
                A[counteq, nbus * 3 + val_bus['idx']] = 1
                A[counteq, nbus * 3 + nbus * 6 + nbranch * 6 + val_bus['idx']] = val_bus['injection'][0].real * baseS
                A[counteq, nbus * 3 + nbus * 6 + nbranch * 6 + nbus + val_bus['idx']] = -val_bus['pv'][0].real * baseS
                b[counteq] = val_bus['injection'][0].real * baseS - val_bus['pv'][0].real * baseS + val_bus['pq'][0].real * baseS
                counteq += 1

                A[counteq, nbus * 4 + val_bus['idx']] = 1
                A[counteq, nbus * 3 + nbus * 6 + nbranch * 6 + val_bus['idx']] = val_bus['injection'][1].real * baseS
                A[counteq, nbus * 3 + nbus * 6 + nbranch * 6 + nbus + val_bus['idx']] = -val_bus['pv'][1].real * baseS
                b[counteq] = val_bus['injection'][1].real * baseS - val_bus['pv'][1].real * baseS + val_bus['pq'][1].real * baseS
                counteq += 1

                A[counteq, val_bus['idx'] + nbus * 5] = 1
                A[counteq, nbus * 3 + nbus * 6 + nbranch * 6 + val_bus['idx']] = val_bus['injection'][2].real * baseS
                A[counteq, nbus * 3 + nbus * 6 + nbranch * 6 + nbus + val_bus['idx']] = -val_bus['pv'][2].real * baseS
                b[counteq] = val_bus['injection'][2].real * baseS - val_bus['pv'][2].real * baseS + val_bus['pq'][2].real * baseS
                counteq += 1

        # Constraints for alphas
        for k in range(nbus * 2):
            G[countineq, nbus * 3 + nbus * 6 + nbranch * 6 + k] = -1
            h[countineq] = 0.0
            countineq += 1

            G[countineq, nbus * 3 + nbus * 6 + nbranch * 6 + k] = 1
            h[countineq] = 1.0
            countineq += 1

        # Constraint 2: Vj = Vi - Zij Sij* - Sij Zij*
        basez = 4.16 ** 2 / 100
        def voltage_cons(A, b, p, frm, to, counteq, pii, qii, pij, qij, pik, qik):
            A[counteq, frm] = 1
            A[counteq, to] = -1
            # real power drop
            A[counteq, p + nbus * 9] = pii / basez
            A[counteq, p + nbus * 9 + nbranch] = pij / basez
            A[counteq, p + nbus * 9 + nbranch * 2] = pik / basez
            # reactive power drop
            A[counteq, p + nbus * 9 + nbranch * 3] = qii / basez
            A[counteq, p + nbus * 9 + nbranch * 4] = qij / basez
            A[counteq, p + nbus * 9 + nbranch * 5] = qik / basez
            b[counteq] = 0.0
            return A, b

        # Write the voltage constraints for connected branches only
        # print("Formulating voltage constraints")
        idx = 0
        v_lim = []
        for k, val_br in branch_sw_data_case.items():
            # Not writing voltage constraints for transformers
            if val_br['type'] != 'XFMR1':
                z = val_br['zprim']
                v_lim.append(val_br['from'])
                v_lim.append(val_br['to'])
                # Writing three phase voltage constraints
                # Phase A
                paa, qaa = -2 * z[0, 0].real, -2 * z[0, 0].imag
                pab, qab = -(- z[0, 1].real + math.sqrt(3) * z[0, 1].imag), -(
                            - z[0, 1].imag - math.sqrt(3) * z[0, 1].real)
                pac, qac = -(- z[0, 2].real - math.sqrt(3) * z[0, 2].imag), -(
                            - z[0, 2].imag + math.sqrt(3) * z[0, 2].real)
                A, b = voltage_cons(A, b, idx, val_br['from'], val_br['to'], counteq, paa, qaa, pab, qab, pac, qac)
                counteq += 1
                # Phase B
                pbb, qbb = -2 * z[1, 1].real, -2 * z[1, 1].imag
                pba, qba = -(- z[0, 1].real - math.sqrt(3) * z[0, 1].imag), -(
                            - z[0, 1].imag + math.sqrt(3) * z[0, 1].real)
                pbc, qbc = -(- z[1, 2].real + math.sqrt(3) * z[1, 2].imag), -(
                            - z[1, 2].imag - math.sqrt(3) * z[1, 2].real)
                A, b = voltage_cons(A, b, idx, nbus + val_br['from'], nbus + val_br['to'], counteq, pba, qba, pbb, qbb,
                                    pbc, qbc)
                counteq += 1
                # Phase C
                pcc, qcc = -2 * z[2, 2].real, -2 * z[2, 2].imag
                pca, qca = -(- z[0, 2].real + math.sqrt(3) * z[0, 2].imag), -(
                            - z[0, 2].imag - math.sqrt(3) * z[0, 2].real)
                pcb, qcb = -(- z[1, 2].real - math.sqrt(3) * z[1, 2].imag), -(
                            - z[0, 2].imag + math.sqrt(3) * z[1, 2].real)
                A, b = voltage_cons(A, b, idx, nbus * 2 + val_br['from'], nbus * 2 + val_br['to'], counteq, pca, qca,
                                    pcb, qcb, pcc, qcc)
                counteq += 1
            idx += 1

        # The node after substation transformer is where we start the voltage constraints
        A[counteq, agent_bus_idx] = 1
        b[counteq] = vsrc[0] ** 2
        counteq += 1
        A[counteq, agent_bus_idx + nbus] = 1
        b[counteq] = vsrc[1] ** 2
        counteq += 1
        A[counteq, agent_bus_idx + nbus * 2] = 1
        b[counteq] = vsrc[2] ** 2
        counteq += 1

        # Constraint 3: 0.95^2 <= V <= 1.05^2 (For those nodes where voltage constraint exist)
        # print("Formulating voltage limit constraints")
        v_idxs = list(set(v_lim))
        # TODO: Does the vmin make sense here?
        vmax = 1.05
        vmin = 0.95
        # print(v_idxs)
        # countineq = 0
        for k in range(nbus):
            if k in v_idxs:
                # Upper bound
                G[countineq, k] = 1
                h[countineq] = vmax ** 2
                countineq += 1
                G[countineq, k+nbus] = 1
                h[countineq] = vmax ** 2
                countineq += 1
                G[countineq, k+nbus*2] = 1
                h[countineq] = vmax ** 2
                countineq += 1
                # Lower Bound
                G[countineq, k] = -1
                h[countineq] = - (vmin) ** 2
                countineq += 1
                G[countineq, k + nbus] = -1
                h[countineq] = - (vmin) ** 2
                countineq += 1
                G[countineq, k + nbus * 2] = -1
                h[countineq] = - (vmin) ** 2
                countineq += 1

        x = cp.Variable(n)
        # print("Calling solver and solving the optimization problem")
        # prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
        #                 [G @ x <= h,
        #                 A @ x == b])
        prob = cp.Problem(cp.Minimize(q.T @ x),
                          [G @ x <= h,
                          A @ x == b])
        prob.solve(solver=cp.ECOS, verbose=False, max_iters=1000, abstol=1e-4)
        # print(prob.status)
        if prob.status == 'infeasible':
            print("Check for limits. Power flow didn't converge")
            return 0

        # Print result.
        # print("\nThe optimal value is", (prob.value))

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
        for k in range(nbus * 9, nbus * 9 + nbranch):
            flow.append([name[i], from_bus[i], to_bus[i], '{:.3f}'.format(x.value[k] * mul),
                         '{:.3f}'.format(x.value[k + nbranch] * mul), \
                         '{:.3f}'.format(x.value[k + nbranch * 2] * mul),
                         '{:.3f}'.format(x.value[k + nbranch * 3] * mul),
                         '{:.3f}'.format(x.value[k + nbranch * 4] * mul), \
                         '{:.3f}'.format(x.value[k + nbranch * 5] * mul)])
            i += 1
        # print(tabulate(flow, headers=['Line Name', 'from', 'to', 'P_A', 'P_B', 'P_C', 'Q_A', 'Q_B', 'Q_C'],
        #                tablefmt='psql'))

        name = []
        for key, val_br in bus_info.items():
            name.append(key)
            # print(key, val_br['idx'])
        # print('\n Voltages at buses:')
        volt = []
        bus_voltage = {}
        try:
            for k in range(nbus):
                # print(name[k])
                # volt.append([name[k], '{:.3f}'.format((x.value[k])), '{:.3f}'.format((x.value[nbus+k])), '{:.3f}'.format((x.value[nbus*2+k])), \
                # '{:.3f}'.format(math.sqrt(abs(x.value[k]))/2401.77), '{:.3f}'format(math.sqrt(abs(x.value[k]))/2401.77), '{:.3f}'format(math.sqrt(abs(x.value[k]))/2401.77)])
                volt.append(
                    [name[k], '{:.3f}'.format(math.sqrt(x.value[k])), '{:.3f}'.format(math.sqrt(x.value[nbus + k])),
                     '{:.3f}'.format(math.sqrt(x.value[nbus * 2 + k])), \
                     '{:.3f}'.format(math.sqrt(abs(x.value[k])) / 2401.77),
                     '{:.3f}'.format(math.sqrt(abs(x.value[nbus + k])) / 2401.77),
                     '{:.3f}'.format(math.sqrt(abs(x.value[nbus * 2 + k])) / 2401.77)])
                # volt.append([name[k], '{:.3f}'.format(math.sqrt(x.value[k])), '{:.3f}'.format(math.sqrt(x.value[nbus+k])), '{:.3f}'.format(math.sqrt(x.value[nbus*2+k]))])
                bus_voltage[name[k]] = {}
                bus_voltage[name[k]]['A'] = math.sqrt(x.value[k])
                bus_voltage[name[k]]['B'] = math.sqrt(x.value[nbus + k])
                bus_voltage[name[k]]['C'] = math.sqrt(x.value[nbus * 2 + k])
            # print(tabulate(volt, headers=['Bus Name', 'V_A', 'V_B', 'V_C', 'V_A (pu)', 'V_B (pu)', 'V_C (pu)'],
                        #    tablefmt='psql'))
        except:
            pass
        # print('\n Injections at buses:')
        injection = []
        bus_injection = {}
        for k in range(nbus):
            injection.append([name[k], '{:.3f}'.format((x.value[k+ nbus*3])*mul), '{:.3f}'.format((x.value[nbus*4+k]) * mul), \
            '{:.3f}'.format((x.value[nbus*5+k])*mul), '{:.3f}'.format(x.value[nbus * 3 + nbus * 6 + nbranch * 6 + k]), '{:.3f}'.format(x.value[nbus * 3 + nbus * 6 + nbranch * 6 + nbus + k])])
            bus_injection[name[k]] = {}
            bus_injection[name[k]]['A'] = (x.value[k + nbus*3]) * mul
            bus_injection[name[k]]['B'] = (x.value[nbus*4+k]) * mul
            bus_injection[name[k]]['C'] = (x.value[nbus*5+k]) * mul
        # print(tabulate(injection, headers=['Bus Name', 'P_Ainj', 'P_Binj', 'P_Cinj', 'Alpha_Pinj', 'Alpha_PV'], tablefmt='psql'))

        # sum = 0.0
        # for i in range(nbranch * 100 + nbus * 6):
        #     s = 0.0
        #     for k in range(nbus * 3  + nbus * 6 + nbranch * 6 + nbus):
        #         s += A[i, k] * x.value[k]
        #     sum += s - b[i]
        #     #print(s, b[i])

        # print("\n The Ax-b expression sum is:", sum, "\n")

        objective = (prob.value)
        status = prob.status
        return bus_voltage