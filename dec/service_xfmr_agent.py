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


class Secondary_Agent(object):
    """
    asdf
    """
    
    def __init__(self):
        self.name = 'area'
        
    
    # Optimization to regulate the voltage
    def alpha_area(self, xfmr_tpx, bus_info, agent_bus, agent_bus_idx, vsrc):


        # Forming the optimization variables
        nbranch = len(xfmr_tpx)
        nbus = len(bus_info)
        # Number of decision variables
        n = nbus   + nbus * 2 + nbranch * 2 + nbus
        # Number of equality/inequality constraints (Injection equations (ABC) at each bus)
        p = nbranch * 100 + nbus * 6
        
        # Initialize the matrices
        P = np.random.randn(n, n) * 0
        q = np.random.randn(n) * 0
        A = np.zeros((p, n))
        b = np.zeros(p)  
        G = np.zeros((p, n))
        h = np.zeros(p)
    
        P = P.T @ P
        # The objective function is written inside P
        # TODO How will the maximize problem look like
        for keyb, val_bus in bus_info.items():
            P[nbus * 1  + nbus * 2 + nbranch * 2 + val_bus['idx'], nbus * 1  + nbus * 2 + nbranch * 2 + val_bus['idx']] = 1
            q[nbus * 1  + nbus * 2 + nbranch * 2 + val_bus['idx']] = -1
        # Define the constraints
        # Constraint 1: sum(Sij) - sum(Sjk) == -sj
        def power_balance (A, b, k_frm, k_to, counteq, col, val):
            for k in k_frm:
                A[counteq, col+k] = -1
            for k in k_to:
                A[counteq, col+k] = 1
            
            A[counteq, val] = -1
            b[counteq] = 0
            return A, b

        # Define it for both real and reactive power
        counteq = 0
        baseS = 1 / (50000)
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
                # Phase 1
                A, b = power_balance(A, b, k_frm, k_to, counteq, nbus*3, val_bus['idx'] + nbus*1 )
                counteq +=1

                # Reactive Power balance equations
                # Phase 1
                A, b = power_balance(A, b, k_frm, k_to, counteq, nbus*3+nbranch, val_bus['idx'] + nbus*2)
                counteq +=1
                
            
        # Make injection a decision variable
        countineq = 0
        for keyb, val_bus in bus_info.items():
            if agent_bus != keyb:
                # Real power injection at a bus
                A[counteq, nbus * 1 + val_bus['idx']] = 1
                A[counteq, nbus * 1  + nbus * 2 + nbranch * 2 + val_bus['idx']] = -val_bus['injection'][0].real*baseS
                b[counteq] = 0
                counteq +=1 

                # Reactive power injection at a bus
                A[counteq, nbus * 2 + val_bus['idx']] = 1
                A[counteq, nbus * 1  + nbus * 2 + nbranch * 2 + val_bus['idx']] = -val_bus['injection'][0].imag*baseS
                b[counteq] = 0
                counteq +=1 

                # Inequality constraints for injection
                G[countineq, nbus * 1  + nbus * 2 + nbranch * 2 + val_bus['idx']] = 1
                h[countineq] = 1.0
                countineq += 1
                # G[countineq, nbus * 3  + nbus * 6 + nbranch * 6 + val_bus['idx']] = -1
                # h[countineq] = -0.5
                # countineq += 1
                
        # Constraint 2: Vj = Vi - Zij Sij* - Sij Zij*
        def voltage_cons (A, b, p, frm, to, counteq, p_pri, q_pri, p_sec, q_sec):
            A[counteq, frm] = 1
            A[counteq, to] = -1
            # real power drop
            A[counteq, p+nbus*3] = p_pri + 0.5 *p_sec
            
            # reactive power drop
            A[counteq, p+nbus*3+nbranch] = q_pri + 0.5 * q_sec
            
            b[counteq] = 0.0
            return A, b
        
        # Write the constraints for connected branches only
        idx = 0
        v_lim = []
        for k, val_br in xfmr_tpx.items():
            # For split phase transformer, we use interlace design
            if val_br['type'] == 'XFMR':
                zp = val_br['impedance']
                zs = val_br['impedance1']
                v_lim.append(val_br['from'])
                v_lim.append(val_br['to'])
                # Writing voltage constraints
                # Phase 1
                p_pri, q_pri = -2 * zp[0].real, -2 * zp[0].imag
                p_sec, q_sec = -2 * zs[0].real, -2 * zs[0].imag
                A, b = voltage_cons(A, b, idx, val_br['from'], val_br['to'], counteq, p_pri, q_pri, p_sec, q_sec)
                counteq += 1
            
            # For triplex line, we assume there is no mutual coupling
            if val_br['type'] != 'XFMR':
                zp = val_br['impedance']
                v_lim.append(val_br['from'])
                v_lim.append(val_br['to'])
                # Writing voltage constraints
                # Phase 1
                p_s1, q_s1 = 0, 0
                p_s2, q_s2 = -2 * zp[0].real, -2 * zp[0].imag
                A, b = voltage_cons(A, b, idx, val_br['from'], val_br['to'], counteq, p_s1, q_s1, p_s2, q_s2)
                counteq += 1
            idx += 1
                
        # The node where service transformer is where we start the voltage constraints
        A[counteq, agent_bus_idx] = 1
        b[counteq] = vsrc[0] ** 2
        counteq += 1

        # Constraint 3: 0.95^2 <= V <= 1.05^2 (For those nodes where voltage constraint exist)
        v_idxs = list(set(v_lim))
        #print(v_idxs)
        for k in range(nbus):
            if k in v_idxs:
                # Upper bound
                G[countineq, k] = 1
                h[countineq] = (1.05) ** 2
                countineq += 1
                # Lower Bound
                G[countineq, k] = -1
                h[countineq] = -(0.95) ** 2
                countineq += 1

        # constant_term = v_meas[0]**2 + v_meas[1]**2 + pmeas[0]**2 + pmeas[1]**2
        x = cp.Variable(n)
        
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x ),
                        [G @ x <= h,
                        A @ x == b])
        # prob = cp.Problem(cp.Minimize((1)*cp.quad_form(x, P) + q.T @ x),
        #             [A @ x == b])
        # prob.solve(verbose=True)
        prob.solve(solver=cp.ECOS, verbose=True)

        # Print result.
        print("\nThe optimal value is", (prob.value))
       
        # Printing the line flows
        from_bus = []
        to_bus = []
        name = []
        for key, val_br in xfmr_tpx.items():
            from_bus.append(val_br['fr_bus'])
            to_bus.append(val_br['to_bus'])
            name.append(key)
        print('\n Real and Reactive Power flow:')
        i = 0
        flow = []
        mul = 1/(baseS * 1000)
        for k in range(nbus*3, nbus*3+nbranch):
            flow.append([name[i], from_bus[i], to_bus[i], '{:.3f}'.format(x.value[k]*mul), '{:.3f}'.format(x.value[k+nbranch]*mul)])
            i += 1
        print(tabulate(flow, headers=['Line Name', 'from', 'to', 'P_s1s2', 'Q_s1s2'], tablefmt='psql'))
        
        name = []
        for key, val_br in bus_info.items():
            name.append(key)
        print('\n Voltages at buses:')
        volt = []
        bus_voltage = {}
        try:
            for k in range(nbus):
                volt.append([name[k], '{:.4f}'.format(math.sqrt(x.value[k]))])
            print(tabulate(volt, headers=['Bus Name', 'V_S1'], tablefmt='psql'))
        except:
            pass
        
        print('\n Injections at buses:')
        injection = []
        for k in range(nbus):
            injection.append([name[k], '{:.3f}'.format((x.value[k+ nbus])*mul), '{:.3f}'.format((x.value[k+ nbus*3 +nbranch*2]))])
        print(tabulate(injection, headers=['Bus Name', 'P_inj', 'Alpha'], tablefmt='psql'))

        sum = 0.0
        for i in range(nbranch * 100 + nbus * 6):
            s = 0.0
            for k in range(nbus * 1  + nbus * 2 + nbranch * 2 + nbus):
                s += A[i, k] * x.value[k] 
            sum += s - b[i]
            #print(s, b[i])  
            
        print("\n The Ax-b expression sum is:", sum, "\n")

        objective = (prob.value)
        status = prob.status
        return objective, status