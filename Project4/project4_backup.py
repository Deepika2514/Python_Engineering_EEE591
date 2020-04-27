# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 00:46:34 2020

@author: Deepika
"""



import numpy as np
import shutil
import subprocess
import warnings
warnings.filterwarnings("ignore")
np.seterr(divide = 'ignore')

# list all fanout values for test
fan = [2]
#fan = [2,3,4,5,6,7]
# list all num of inverters for test
inv_stage = [3]
#inv_stage = [3,5,7,9,11,13]


# list all possible (fanout & stage) combinations (each pair as a tuple)
combinations = []
for f in fan:
    for inv in inv_stage:
        comb = tuple([f, inv]) # each config = tuple(fanout, stage)
        combinations.append(comb)

# a list to store all results of delay time
result = []
node = 98
for counter in range(len(combinations)):
    # extract fanout & inverter nums of current test run
    fan_new = combinations[counter][0]
    stage_new = combinations[counter][1]
    # My code
    with open("header.sp") as f:
        with open("InvChain_temp.sp", "w+") as f1:
            for line in f:
                f1.write(line)
				
    with open("InvChain_temp.sp", "a") as file:
        line_fan_param = ".param fan = " + str(fan_new) + "\n"
        file.write(line_fan_param)
		##writing fans line
        line = "Xinv1 a " + chr(node) + " inv M = 1\n"
        file.write(line)
	
		# write 2nd ~ last-1 stages of inverter
        for j in np.arange(1, stage_new-1 , 1):
			# set fanout multiplies for each stage
            fans = "**" + str(j)
            line = "Xinv" + str(j + 1) + " " + chr(j + node - 1) + ' ' + chr(j + node) + " inv M = fan" + fans + "\n"
            
            file.write(line)
	
		# write the last stage, always ends with node "z",
		# and set the fanout multiplies
        fans = "**" +str(j+1)
        line = "Xinv"+ str(stage_new) + " " + chr(j + node) + " z inv M = fan" + fans + "\n"
        file.write(line)
			
		##Writing end
        line_end = ".end\n"
        file.write(line_end)
	
	# overwrite the original script file with temp file
    shutil.copyfile('InvChain_temp.sp', 'InvChain.sp')


    # run hspice
    proc = subprocess.Popen(["hspice", "InvChain.sp"], stdout = subprocess.PIPE)
    output, err = proc.communicate()
    #print("*** Running hspice InvChain.sp command ***\n", output)

    # extract delay time from output csv
    Data = np.recfromcsv("InvChain.mt0.csv", comments = "$", skip_header = 3)
    #print(Data["tphl_inv"])
    # store delay time for each config, use [()] to extract the float num from
    # an np.array with no dimension
    result.append(Data["tphl_inv"][()])
             


	
# test run finished, find the minimum delay time from result list
opt_val = min(result)
# use minimum delay time to find the corresponding config
opt_config = combinations[result.index(opt_val)]

# print all combinations of (fanout & num of inverters) and corresponding delay time 
print("fan inverter  delay(sec)")
for i in range(len(combinations)):
    print("{0:3d} {1:8d}  {2:6.4e}".format(combinations[i][0], combinations[i][1], result[i]))

# print minumum delay time & its corresponding config
print("\nminimum delay (sec): {0:.4e}".format(opt_val))
print("optimum config => fan = {0:d}; inverter = {1:d}".format(opt_config[0], opt_config[1]))
			

