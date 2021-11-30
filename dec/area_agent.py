# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:11:21 2021

@author: poud579
"""

import numpy as np
from itertools import product
import networkx as nx
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
    def alpha_area(self, branch_sw_data_case,  bus_info):

        # Choose the system to run power flow    
        nbranch = len(branch_sw_data_case)
        nbus = len(bus_info)
        
        # Finding number of variables for optimization. Flow constraints/injection
        # at all buses and voltage constraints along each branches
            
        # Find source injection beforehand for writing flow constraints
        sourceinj = np.zeros((3), dtype=complex)
        for b in bus_info.values():
            sourceinj[0] -= b['injection'][0]
            sourceinj[1] -= b['injection'][1]
            sourceinj[2] -= b['injection'][2]
        bus_info['150']['injection'] = sourceinj

        # Forming the optimization variables
        m = nbus * 6
        # Number of decision variables
        n = nbus * 3  + nbranch * 6
        # Number of equality constraints (Injection equations (ABC) at each bus)
        p = nbranch * 10 + nbus * 6
        
        # Initialize the matrices
        P = np.random.randn(n, n) * 0
        q = np.random.randn(n) * 0
        A = np.zeros((p, n))
        b = np.zeros((p))  
    
        P = P.T @ P
        # print(P)
        # print(q)

        # Define the constraints
        # Constraint 1: sum(Sij) - sum(Sjk) == -sj
        def power_balance (A, b, k_frm, k_to, counteq, col, val):
            for k in k_frm:
                A[counteq, col+k] = -1
            for k in k_to:
                A[counteq, col+k] = 1
            b[counteq] = val
            return A, b
        
        # Define it for both real and reactive power
        counteq = 0
        for key, val_bus in bus_info.items():
            k_frm = []
            k_to = []
            # Find bus idx in "from" of branch_sw_data
            ind_frm = 0
            ind_to = 0
            for key, val_br in branch_sw_data_case.items():
                #print(key, val_br['from'], val_br['to'])
                if val_bus['idx'] == val_br['from']:
                    k_frm.append(ind_frm)
                if val_bus['idx'] == val_br['to']:
                    k_to.append(ind_to)
                ind_to += 1
                ind_frm += 1
            # Real Power balance equations
            # Phase A
            A, b = power_balance(A, b, k_frm, k_to, counteq, nbus*3, val_bus['injection'][0].real)
            counteq +=1
            # Phase B
            A, b = power_balance(A, b, k_frm, k_to, counteq, nbus*3+nbranch, val_bus['injection'][1].real)
            counteq +=1
            # Phase C
            A, b = power_balance(A, b, k_frm, k_to, counteq, nbus*3+nbranch*2, val_bus['injection'][2].real)
            counteq +=1 
            
            # Reactive Power balance equations
            # Phase A
            A, b = power_balance(A, b, k_frm, k_to, counteq, nbus*3+nbranch*3, val_bus['injection'][0].imag)
            counteq +=1
            # Phase B
            A, b = power_balance(A, b, k_frm, k_to, counteq, nbus*3+nbranch*4, val_bus['injection'][1].imag)
            counteq +=1
            # Phase C
            A, b = power_balance(A, b, k_frm, k_to, counteq, nbus*3+nbranch*5, val_bus['injection'][2].imag)
            counteq +=1 

        # Constraint 2: Vj = Vi - Zij Sij* - Sij Zij*
        def voltage_cons (A, b, p, frm, to, counteq, pii, qii, pij, qij, pik, qik):
            A[counteq, frm] = 1
            A[counteq, to] = -1
            # real power drop
            A[counteq, p+nbus*3] = pii 
            A[counteq, p+nbus*3+nbranch] = pij 
            A[counteq, p+nbus*3+nbranch*2] = pik 
            # reactive power drop
            A[counteq, p+nbus*3+nbranch*3] = qii 
            A[counteq, p+nbus*3+nbranch*4] = qij 
            A[counteq, p+nbus*3+nbranch*5] = qik 
            b[counteq] = 0.0
            return A, b
        
        # Write the constraints for connected branches only
        idx = 0
        v_lim = []
        for k, val_br in branch_sw_data_case.items():
            # Not writing voltage constraints for transformers
            if val_br['type'] != 'XFMR':
                z = val_br['zprim']
                v_lim.append(val_br['from'])
                v_lim.append(val_br['to'])
                # Writing three phase voltage constraints
                # Phase A
                paa, qaa = -2 * z[0,0].real, -2 * z[0,0].imag
                pab, qab = -(- z[0,1].real + math.sqrt(3) * z[0,1].imag), -(- z[0,1].imag - math.sqrt(3) * z[0,1].real)
                pac, qac = -(- z[0,2].real - math.sqrt(3) * z[0,2].imag), -(- z[0,2].imag + math.sqrt(3) * z[0,2].real)
                A, b = voltage_cons(A, b, idx, val_br['from'], val_br['to'], counteq, paa, qaa, pab, qab, pac, qac)
                counteq += 1
                # Phase B
                pbb, qbb = -2 * z[1,1].real, -2 * z[1,1].imag
                pba, qba = -(- z[0,1].real - math.sqrt(3) * z[0,1].imag), -(- z[0,1].imag + math.sqrt(3) * z[0,1].real)
                pbc, qbc = -(- z[1,2].real + math.sqrt(3) * z[1,2].imag), -(- z[1,2].imag - math.sqrt(3) * z[1,2].real)
                A, b = voltage_cons(A, b, idx, nbus+val_br['from'], nbus+val_br['to'], counteq, pba, qba, pbb, qbb, pbc, qbc)
                counteq += 1
                # Phase C
                pcc, qcc = -2 * z[2,2].real, -2 * z[2,2].imag
                pca, qca = -(- z[0,2].real + math.sqrt(3) * z[0,2].imag), -(- z[0,2].imag - math.sqrt(3) * z[0,2].real)
                pcb, qcb = -(- z[1,2].real - math.sqrt(3) * z[1,2].imag), -(- z[0,2].imag + math.sqrt(3) * z[1,2].real)
                A, b = voltage_cons(A, b, idx, nbus*2+val_br['from'], nbus*2+val_br['to'], counteq, pca, qca, pcb, qcb, pcc, qcc)
                counteq += 1
            idx += 1
        
        # The node after substation transformer is where we start the voltage constraints
        A[counteq, 11] = 1
        b[counteq] = 2401.77 ** 2
        counteq += 1
        A[counteq, 11+nbus] = 1
        b[counteq] = 2401.77 ** 2
        counteq += 1
        A[counteq, 11+nbus*2] = 1
        b[counteq] = 2401.77 ** 2
        counteq += 1
        
        # Constraint 3: 0.95^2 <= V <= 1.05^2 (For those nodes where voltage constraint exist)
        G   = np.zeros((nbus*6, n))
        h   = np.zeros(nbus*6)
        v_idxs = list(set(v_lim))
        #print(v_idxs)
        countineq = 0
        # for k in range(nbus):
        #     if k in v_idxs:
        #         # Upper bound
        #         G[countineq, k] = 1
        #         h[countineq] = (1.05 * 2401.77) ** 2
        #         countineq += 1
        #         G[countineq, k+nbus] = 1
        #         h[countineq] = (1.05 * 2401.77) ** 2
        #         countineq += 1
        #         G[countineq, k+nbus*2] = 1
        #         h[countineq] = (1.05 * 2401.77) ** 2
        #         countineq += 1
                # Lower Bound
                # G[countineq, k] = -1
                # h[countineq] = -(0.8 * 2401.77) ** 2
                # countineq += 1
                # G[countineq, k+nbus] = -1
                # h[countineq] = -(0.8 * 2401.77) ** 2
                # countineq += 1
                # G[countineq, k+nbus*2] = -1
                # h[countineq] = -(0.8 * 2401.77) ** 2
                # countineq += 1


        # constant_term = v_meas[0]**2 + v_meas[1]**2 + pmeas[0]**2 + pmeas[1]**2
        x = cp.Variable(n)
        
        prob = cp.Problem(cp.Minimize((1)*cp.quad_form(x, P) + q.T @ x ),
                        [G @ x <= h,
                        A @ x == b])
        # prob = cp.Problem(cp.Minimize((1)*cp.quad_form(x, P) + q.T @ x + constant_term),
        #             [A @ x == b])
        prob.solve(verbose=True, max_iter=50000)

        # Print result.
        print("\nThe optimal value is", (prob.value))
       
        # Printing the line flows
        from_bus = []
        to_bus = []
        name = []
        for key, val_br in branch_sw_data_case.items():
            from_bus.append(val_br['from'])
            to_bus.append(val_br['to'])
            name.append(key)
        print('\n Real and Reactive Power flow:')
        i = 0
        flow = []
        for k in range(nbus*3, nbus*3+nbranch):
            flow.append([name[i], from_bus[i], to_bus[i], '{:.2f}'.format(x.value[k]/1000), '{:.2f}'.format(x.value[k+nbranch]/1000), \
            '{:.2f}'.format(x.value[k+nbranch*2]/1000), '{:.2f}'.format(x.value[k+nbranch*3]/1000), '{:.2f}'.format(x.value[k+nbranch*4]/1000), '{:.2f}'.format(x.value[k+nbranch*5]/1000)])
            i += 1
        print(tabulate(flow, headers=['Name', 'from', 'to', 'P_A', 'P_B', 'P_C', 'Q_A', 'Q_B', 'Q_C'], tablefmt='psql'))
        
        name = []
        for key, val_br in bus_info.items():
            name.append(key)
        print('\n Voltages at buses:')
        volt = []
        bus_voltage = {}
        try:
            for k in range(nbus):
                # volt.append([name[k], '{:.2f}'.format((x.value[k])), '{:.2f}'.format((x.value[nbus+k])), '{:.2f}'.format((x.value[nbus*2+k])), \
                # '{:.3f}'.format(math.sqrt(abs(x.value[k]))/2401.77), '{:.3f}'format(math.sqrt(abs(x.value[k]))/2401.77), '{:.3f}'format(math.sqrt(abs(x.value[k]))/2401.77)])
                volt.append([name[k], '{:.2f}'.format(math.sqrt(x.value[k])), '{:.2f}'.format(math.sqrt(x.value[nbus+k])), '{:.2f}'.format(math.sqrt(x.value[nbus*2+k])), \
                '{:.3f}'.format(math.sqrt(abs(x.value[k]))/2401.77), '{:.3f}'.format(math.sqrt(abs(x.value[nbus+k]))/2401.77), '{:.3f}'.format(math.sqrt(abs(x.value[nbus*2+k]))/2401.77)])
                # volt.append([name[k], '{:.2f}'.format(math.sqrt(x.value[k])), '{:.2f}'.format(math.sqrt(x.value[nbus+k])), '{:.2f}'.format(math.sqrt(x.value[nbus*2+k]))])
                bus_voltage[name[k]] = {}
                bus_voltage[name[k]]['A'] = math.sqrt(x.value[k])
                bus_voltage[name[k]]['B'] = math.sqrt(x.value[nbus+k])
                bus_voltage[name[k]]['C'] = math.sqrt(x.value[nbus*2+k])
            print(tabulate(volt, headers=['Name', 'V_A', 'V_B', 'V_C', 'V_A (pu)', 'V_B (pu)', 'V_C (pu)'], tablefmt='psql'))
        except:
            pass
        
        sum = 0.0
        for i in range(nbranch * 10 + nbus * 6):
            s = 0.0
            for k in range(nbus * 3  + nbranch * 6):
                s += A[i, k] * x.value[k] 
            sum += s - b[i]
            #print(s, b[i])  
            
        print("\n The Ax-b expression sum is:", sum, "\n")

        objective = (prob.value)
        status = prob.status
        return objective, status
