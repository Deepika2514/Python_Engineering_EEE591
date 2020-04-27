################################################################################
# Project 4                                                                    #
# Steve Millman                                                                #
# Run hspice to determine the tphl of a circuit                                #
################################################################################

import numpy as np      # package needed to read the results file
import subprocess       # package needed to lauch hspice

################################################################################
# Start the main program here.                                                 #
################################################################################

# launch hspice. Note that both stdout and stderr are captured so
# they do NOT go to the terminal!
proc = subprocess.Popen(["hspice","InvChain.sp"],
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE)
output, err = proc.communicate()

# extract tphl from the output file
data = np.recfromcsv("InvChain.mt0.csv",comments="$",skip_header=3)
tphl = data["tphl_inv"]
print(tphl)
