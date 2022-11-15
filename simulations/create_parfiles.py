import math
import itertools

'''
Create files with input parameters of simulations
We define min and max of beam size and divergence and then use these valus to define min and max of emiitance and beta parameters.
Resulting parameter space is divided on a given number(n_intervals) of cells, and values at their boarders become training/validation configuration. In order to archieve different training and validation data, number of divisions(n_intervals) should be chosen different.
'''

#             min             max
beam_size = [0.5*10**(-6), 10.*10**(-6)] #meters
divergence = [0.5*10**(-3), 3.*10**(-3)] #rad

emmitance = [beam_size[0]*divergence[0], beam_size[1]*divergence[1]]
beta = [beam_size[0]/divergence[0], beam_size[1]/divergence[1]]

print('Range of emmitance: ')
print(beam_size[0]*divergence[0], beam_size[1]*divergence[1])

print('Range of beta: ')
print(beam_size[0]/divergence[1], beam_size[1]/divergence[0])

#training
n_intervals = 4
emmitances_tr = [(emmitance[-1]-emmitance[0])/(n_intervals-1)*i+emmitance[0] for i in range(n_intervals)][1:-1]
betas_tr = [(beta[-1]-beta[0])/(n_intervals-1)*i+beta[0] for i in range(n_intervals)][1:-1]
print(emmitances_tr)
print(betas_tr)

#validation
n_intervals = 6
emmitances_val = [(emmitance[-1]-emmitance[0])/(n_intervals-1)*i+emmitance[0] for i in range(1, n_intervals-1)][1:-1]
betas_val = [(beta[-1]-beta[0])/(n_intervals-1)*i+beta[0] for i in range(1, n_intervals-1)][1:-1]
print(emmitances_val)
print(betas_val)

pars_tr = itertools.product(emmitances_tr, emmitances_tr, betas_tr, betas_tr)
pars_val = itertools.product(emmitances_val, emmitances_val, betas_val, betas_val)

for p in pars_tr:
    print('Create: ', "./data/elegant_files/train/Track_" + "{:.3e}".format(p[0]) + "_" + 
                                                           "{:.3e}".format(p[1]) + "_" +
                                                           "{:.3e}".format(p[2]) + "_" +
                                                           "{:.3e}".format(p[3]) + ".ele")
    a_file = open("./Track.ele", "r")
    list_of_lines = a_file.readlines()

    list_of_lines[46] = "emit_x = " + str(p[0]) + "\n"
    list_of_lines[47] = "emit_y = " + str(p[1]) + "\n"
    
    list_of_lines[48] = "beta_x = " + str(p[2]) + "\n"
    list_of_lines[49] = "beta_y = " + str(p[3]) + "\n"


    b_file = open("./data/elegant_files/train/Track_" + "{:.3e}".format(p[0]) + "_" + 
                                                           "{:.3e}".format(p[1]) + "_" +
                                                           "{:.3e}".format(p[2]) + "_" +
                                                           "{:.3e}".format(p[3]) + ".ele", "w")
    b_file.writelines(list_of_lines)
    b_file.close()
    a_file.close()

for p in pars_val:
    print('Create: ', "./data/elegant_files/validate/Track_" + "{:.3e}".format(p[0]) + "_" + 
                                                           "{:.3e}".format(p[1]) + "_" +
                                                           "{:.3e}".format(p[2]) + "_" +
                                                           "{:.3e}".format(p[3]) + ".ele")
    a_file = open("./Track.ele", "r")
    list_of_lines = a_file.readlines()

    list_of_lines[46] = "emit_x = " + str(p[0]) + "\n"
    list_of_lines[47] = "emit_y = " + str(p[1]) + "\n"
    
    list_of_lines[48] = "beta_x = " + str(p[2]) + "\n"
    list_of_lines[49] = "beta_y = " + str(p[3]) + "\n"


    b_file = open("./data/elegant_files/validate/Track_" + "{:.3e}".format(p[0]) + "_" + 
                                                           "{:.3e}".format(p[1]) + "_" +
                                                           "{:.3e}".format(p[2]) + "_" +
                                                           "{:.3e}".format(p[3]) + ".ele", "w")
    b_file.writelines(list_of_lines)
    b_file.close()
    a_file.close()