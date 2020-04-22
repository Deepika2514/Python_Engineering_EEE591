# Import packages

from numpy import linspace, nditer, exp, seterr, log10, zeros_like, linalg, asarray
from matplotlib.pyplot import title, grid, plot, xlabel, ylabel, show, legend
import matplotlib.pyplot as plt
from scipy import optimize
seterr(divide = 'ignore')

# global constants 
q = 1.6021766208e-19       # Coulomb's constant 
boltz = 1.380648e-23  # Boltzmann constant

###For Part 1
Is_first = 1e-9  # source current
n_first = 1.7   # ideality
R_first = 11000    # resistor 
T_first = 350   # temperature 

###For part 2
max_tolerance = 1e-4  # maximum allowed tolerance 
max_iteration = 100  # maximum number of allowed iterations
T_second = 375   # temperature
A_second = 1e-8  # sectional-area

# initial guesses for unknown parameters
vd_Init = 0.1      # initial guess for diode voltage value
phi_opt = 0.8     # initial guess for optimal barrier height value
R_opt = 10000      # initial guess for optimal resistor value
n_opt = 1.5       # initial guess for optimal ideality value

def diode_current_calc(vd, Vs):
    # current equation of diode
    diode_current_val = Is_first * (exp((vd * q) / (n_first * boltz * T_first)) - 1)
    # return nodal function = 0
    return ((vd - Vs) / R_first) + diode_current_val
	
def solve_voltage_diode(vd, vs, R, n, T, is_val):    
    # This is the constant in diode current equation
    vt = (n * boltz * T) / q
    # diode current equation
    diode = is_val * (exp(vd / vt) - 1)
    # return nodal function = 0
    return ((vd - vs) / R) + diode	
	
def solve_current_diode(A, phi, R, n, T, v_s):
    # create zero array to store computed diode current/voltage
    diode_voltage_est = zeros_like(v_s)
    current_diode = zeros_like(v_s)
    # specify initial diode voltage for fsolve()
    v_guess = vd_Init
    is_val = A * T * T * exp(-phi * q / ( boltz * T ) )
    
    # for every given source voltage, calculate diode voltage by solving nodal analysis
    for index in range(len(v_s)):
        v_guess = optimize.fsolve(solve_voltage_diode, v_guess, (v_s[index], R, n, T, is_val),
                                xtol = 1e-12)[0]
        diode_voltage_est[index] = v_guess    
    # compute the diode current
    vt = (n * boltz * T) / q  # calc constant for diode current equation
    current_diode = is_val * (exp(diode_voltage_est / vt) - 1) # calc diode current by its definition
    return current_diode
	
def optimize_R(R_guess, phi_guess, n_guess, A, T, v_src, current_meas):  
	# diode current is obtained using optimized parameters
    current_diode = solve_current_diode(A, phi_guess, R_guess, n_guess, T, v_src)
	# Absolute error
    return (current_diode - current_meas)

    
def optimize_phi(phi_guess, R_guess, n_guess, A, T, v_src, current_meas):
	# diode current is obtained using optimized parameters
    current_diode = solve_current_diode(A, phi_guess, R_guess, n_guess, T_second, v_src)
	# Normalized error is obtained by adding a contant in denominator for handling 0/0 case
    return (current_diode - current_meas) / (current_diode + current_meas + 1e-15)
    
def optimize_n(n_guess, R_guess, phi_guess, A, T, v_src, current_meas):
	# diode current is obtained using optimized parameters
    current_diode = solve_current_diode(A, phi_guess, R_guess, n_guess, T_second, v_src)
    # Normalized error is obtained by adding a contant in denominator for handling 0/0 case
    return (current_diode - current_meas) / (current_diode + current_meas + 1e-15)
	
if __name__=="__main__":
    
    #####Part1######
    
    v_src_first = linspace(0.1, 2.5, 25, endpoint = True)  # range of source voltage
    diode_current = []      # to store diode current 
    diode_voltage = [] # to store diode voltage
    
	
    for v in nditer(v_src_first):
        # get diode voltage by solving optimize.fsolve
        vd_Init = optimize.fsolve(diode_current_calc, vd_Init, (v,))[0]
        diode_voltage.append(vd_Init)
        # calculate diode current using diode voltage and diode current equation
        i_d = Is_first * (exp((vd_Init * q) / (n_first * boltz * T_first)) - 1)
        diode_current.append(i_d)
        
    # plotting first part of the project
    print("Problem 1:\n")
    plot(v_src_first, log10(diode_current),"b", label = "Source Voltage")
    plot(diode_voltage, log10(diode_current),"r", label = "diode voltage")
    xlabel("Voltage in volts ", fontsize = 20)
    ylabel("Diode current in log scale", fontsize = 20)
    legend(loc = 'center right')
    grid()
    show()
	
	#####Part2#############

    
    v_src_second = []  # array to store source voltage
    current_meas = []   # array to store measured diode current
    # read datasets into array from file
    filename = "DiodeIV.txt"
    f = open(filename, "r")
    lines = f.readlines()
    for line in lines:
        line = line.strip()  # remove space at the start/end of each line
        if line:
            parameter = line.split(" ")             # split datasets in each line
            v_src_second.append(float(parameter[0])) #  source voltage
            current_meas.append(float(parameter[1]))      #  measured diode current
    v_src_second = asarray(v_src_second)
    current_meas = asarray(current_meas)
	# iteration counter
    itr = 0
    # calculate diode current using initial guesses.It is important to get initial error values array
    current_pred = solve_current_diode(A_second, phi_opt, R_opt, n_opt, T_second, v_src_second)
    # error 
    error = linalg.norm((current_pred - current_meas) / (current_pred + current_meas + 1e-15), ord = 1)#Normalized error
    #print
    print("\n\nProblem 2:")
    print("Iteration No:      R:       Phi:       n:      error:")
	# iterate optimization process until error function is satisfied
    while (error > max_tolerance and itr < max_iteration):
        #for iterations in range(max_iteration):
            # update iteration counter
            itr += 1
            # optimize resistor values for error values array
            R_opt = optimize.leastsq(optimize_R, R_opt, 
                                 args = (phi_opt, n_opt, A_second, T_second, v_src_second, current_meas))[0][0]
            # optimize barrier height values for error values array
            phi_opt = optimize.leastsq(optimize_phi, phi_opt, 
                                   args = (R_opt, n_opt, A_second, T_second, v_src_second, current_meas))[0][0]
            # optimize ideality values for error values array
            n_opt = optimize.leastsq(optimize_n, n_opt, 
                                 args = (R_opt, phi_opt, A_second, T_second, v_src_second, current_meas))[0][0]
            # calc the diode current
            current_pred = solve_current_diode(A_second, phi_opt, R_opt, n_opt, T_second, v_src_second)
            # calc error values array for optimizing result check
            error = linalg.norm((current_pred - current_meas) / (current_pred + current_meas + 1e-15), ord = 1)
            # print the optimized resistor, phi, and ideality values with iteration counter.
            print("{0:9d} {1:7.4f} {2:7.4f} {3:7.4f} {4:7.4f}".format(itr, R_opt, phi_opt, n_opt, error))
    # plot the relationship of source voltage and log10(diode current) after optimization
#    plot(v_src_second, log10(current_meas), "bs-", label = "measured I")
#    plot(v_src_second, log10(current_pred), "r*-", label = "estimated I")
#    xlabel("Source Voltage", fontsize = 20)
#    ylabel("Diode current", fontsize = 20)
#    title("Source voltage vs. Diode current", fontsize = 20)
#    legend(loc = 'center right')
#    grid()
#    show()
    
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('voltage in volts')
    ax1.set_ylabel('measured current in log scale', color=color)
    ax1.plot(v_src_second, log10(current_meas),"bs-", color=color, label = "measured I")
    ax1.tick_params(axis='y', labelcolor=color)
    
    grid()
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('estimated current in log scale', color=color)  # we already handled the x-label with ax1
    ax2.plot(v_src_second, log10(current_pred), "r*-", color=color, label = "estimated I")
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    grid()
    plt.show()
    
    print("optimized value of R: {:.4f}".format(R_opt))
    print("optimized value of ideality: {:.4f}".format(n_opt))
    print("optimized value of phi: {:.4f}".format(phi_opt))
