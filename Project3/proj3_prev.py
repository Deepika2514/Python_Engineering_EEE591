# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:30:48 2020

@author: Deepika
"""

"""
###############################################################################
#                                                                             #
#                                PROJECT 3                                    #
#                             NON-LINEAR DIODE                                #
#                                                                             #
###############################################################################
"""
# Import packages

from numpy import linspace, nditer, exp, seterr, log10, zeros_like, linalg, asarray
from scipy import optimize
from matplotlib.pyplot import title, grid, plot, xlabel, ylabel, show, legend
seterr(divide = 'ignore')

# global constants for the project
q = 1.6021766208e-19       # Coulomb's constant 
boltz = 1.380648e-23  # Boltzmann constant
###For Part 1
Is_1 = 1e-9  # source current
n_1 = 1.7   # ideality
R_1 = 11000    # resistor 
T_1 = 350   # temperature 
###For part 2
max_tolerance = 1e-3  # maximum allowed tolerance 
max_iteration = 100  # maximum number of allowed iterations
T_2 = 375   # temperature
A_2 = 1e-8  # sectional-area
# initial guesses for unknown parameters
Vd_Init = 1.      # initial guess for diode voltage value
phi_opt = 0.8     # initial guess for optimal barrier height value
R_opt = 10000      # initial guess for optimal resistor value
n_opt = 1.5       # initial guess for optimal ideality value
    


def current_diode(volt, Vs):
    # diode current equation
    diode = Is_1 * (exp((volt * q) / (n_1 * boltz * T_1)) - 1)
    # return nodal function = 0
    return ((volt - Vs) / R_1) + diode

def solve_v_diode(vd, vs, R, n, T, is_val):    
    # calc constant for diode current equation
    vt = (n * boltz * T) / q
    # diode current equation
    diode = is_val * (exp(vd / vt) - 1.)
    # return nodal function = 0
    return ((vd - vs) / R) + diode

def solve_i_diode(A, phi, R, n, T, v_src):
    # create zero array to store computed diode current/voltage
    vd_est = zeros_like(v_src)
    i_diode = zeros_like(v_src)
    # specify initial diode voltage for fsolve()
    v_guess = Vd_Init
    is_val = A * T * T * exp(-phi * q / ( boltz * T ) )
    
    # for every given source voltage, calc diode voltage by solving nodal analysis
    for index in range(len(v_src)):
        v_guess = optimize.fsolve(solve_v_diode, v_guess, (v_src[index], R, n, T, is_val),
                                xtol = 1e-12)[0]
        vd_est[index] = v_guess    
    # compute the diode current
    vt = (n * boltz * T) / q  # calc constant for diode current equation
    i_diode = is_val * (exp(vd_est / vt) - 1.) # calc diode current by its definition
    return i_diode

def optimize_R(R_guess, phi_guess, n_guess, A, T, v_src, i_meas):  
    # get diode current using optimized parameters
    i_diode = solve_i_diode(A, phi_guess, R_guess, n_guess, T, v_src)
    # normalized error (add a constant in denominator for avoiding 0/0 case)
    return (i_diode - i_meas)

    
def optimize_phi(phi_guess, R_guess, n_guess, A, T, v_src, i_meas):
    # get diode current using optimized parameters
    i_diode = solve_i_diode(A, phi_guess, R_guess, n_guess, T_2, v_src)
    # normalized error (add a constant in denominator for avoiding 0/0 case)
    return (i_diode - i_meas) / (i_diode + i_meas + 1e-15)
    
def optimize_n(n_guess, R_guess, phi_guess, A, T, v_src, i_meas):
    # get diode current using optimized parameters
    i_diode = solve_i_diode(A, phi_guess, R_guess, n_guess, T_2, v_src)
    # normalized error (add a constant in denominator for avoiding 0/0 case)
    return (i_diode - i_meas) / (i_diode + i_meas + 1e-15)


if __name__=="__main__":
    
    #####Part1######
    
    v_src_1 = linspace(0.1, 2.6, 25, endpoint = True)  # range of source voltage
    diode_current = []      # to store diode current 
    diode_voltage = [] # to store diode voltage

    for v in nditer(v_src_1):
        # get diode voltage by solving f_{node current}(Vd) = 0 
        v_d = optimize.fsolve(current_diode, 1, (v,))[0]
        diode_voltage.append(v_d)
        # calculate diode current using diode voltaage and diode current equation
        i_d = Is_1 * (exp((v_d * q) / (n_1 * boltz * T_1)) - 1.)
        diode_current.append(i_d)
        
    # plot the relationship of source voltage and log10(diode current) 
    print("Problem 1:\n")
    plot(v_src_1, log10(diode_current))
    xlabel("Source Voltage ", fontsize = 20)
    ylabel("Diode current", fontsize = 20)
    title("Source voltage vs Diode current", fontsize = 20)
    grid()
    show()
    
    # plot the relationship of diode voltage and log10(diode current) 
    plot(diode_voltage, log10(diode_current))
    xlabel("Diode Voltage", fontsize = 20)
    ylabel("Diode current", fontsize = 20)
    title("Diode voltage vs. Diode current", fontsize = 20)
    grid()
    show()
    
    
    #####Part2#############

    
    # arrays to store datasets from file
    v_src_2 = []  # source voltage
    i_meas = []   # measured diode current
    # read datasets into array from file
    filename = "DiodeIV.txt"
    fh = open(filename, "r")
    lines = fh.readlines()
    for line in lines:
        line = line.strip()  # remove space at the start/end of each line
        if line:
            parameter = line.split(" ")             # split datasets in each line
            v_src_2.append(float(parameter[0])) #  source voltage
            i_meas.append(float(parameter[1]))      #  measured diode current
    v_src_2 = asarray(v_src_2)
    i_meas = asarray(i_meas)
    # iteration counter
    iteration = 0
    # calculate diode current using initial guesses to get initial error values array
    current2 = solve_i_diode(A_2, phi_opt, R_opt, n_opt, T_2, v_src_2)
    # error 
    error = linalg.norm((current2 - i_meas) / (current2 + i_meas + 1e-15), ord = 1)#Normalized error
    #print
    print("\n\nProblem 2:")
    print("Iteration No:    R:      Phi:      n:")
    # iterate optimization process until error function is satisfied
    while (error > max_tolerance and iteration < max_iteration):
        #for iterations in range(max_iteration):
            # update iteration counter
            iteration += 1
            # optimize resistor values for error values array
            R_opt = optimize.leastsq(optimize_R, R_opt, 
                                 args = (phi_opt, n_opt, A_2, T_2, v_src_2, i_meas))[0][0]
            # optimize barrier height values for error values array
            phi_opt = optimize.leastsq(optimize_phi, phi_opt, 
                                   args = (R_opt, n_opt, A_2, T_2, v_src_2, i_meas))[0][0]
            # optimize ideality values for error values array
            n_opt = optimize.leastsq(optimize_n, n_opt, 
                                 args = (R_opt, phi_opt, A_2, T_2, v_src_2, i_meas))[0][0]
            # calc the diode current
            current2 = solve_i_diode(A_2, phi_opt, R_opt, n_opt, T_2, v_src_2)
            # calc error values array for optimizing result check
            error = linalg.norm((current2 - i_meas) / (current2 + i_meas + 1e-15), ord = 1)
            # print the optimized resistor, phi, and ideality values with iteration counter.
            print("{0:9d} {1:7.2f} {2:7.4f} {3:7.4f}".format(iteration, R_opt, phi_opt, n_opt))
    # plot the relationship of source voltage and log10(diode current) after optimization
    plot(v_src_2, log10(i_meas), "bs-", label = "measured I")
    plot(v_src_2, log10(current2), "r*-", label = "estimated I")
    xlabel("Source Voltage", fontsize = 20)
    ylabel("Diode current", fontsize = 20)
    title("Source voltage vs. Diode current", fontsize = 20)
    legend(loc = 'center right')
    grid()
    show()
    
    print("optimized R: {:.4f}".format(R_opt))
    print("optimized ideality: {:.4f}".format(n_opt))
    print("optimized phi: {:.4f}".format(phi_opt))
