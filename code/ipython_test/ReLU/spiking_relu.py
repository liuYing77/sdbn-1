'''
Typical Usage:
number of clusters per digit = 10
Train digit 0, cluster 0
Simualator: Nest
python train_mnist.py 10 0 0 nest
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import scipy.io as sio
import scipy.cluster.vq as spvq
import scipy.spatial.distance as spdt
import os
import random
import time
import maths_utils as mu

def plot_spikes(spikes, title):
    fig, ax = plt.subplots()
    ax.plot([i[1] for i in spikes], [i[0] for i in spikes], ".")
    plt.show()



sim = sys.argv[1]
if sim == 'nest':
    import pyNN.nest as p
elif sim == 'spin':
    import spynnaker.pyNN as p
else:
    sys.exit()

cell_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 1.,   # 2.0
                   'tau_syn_E': 1.0,
                   'tau_syn_I': 1.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
                   }
  
             
p.setup(timestep=1.0, min_delay=1.0, max_delay=16.0)



run_s = 10.
runtime = 1000. * run_s
max_rate = 1000.
ee_connector = p.OneToOneConnector(weights=1.0, delays=2.0)    


pop_list = []
pop_output = []
pop_source = []
x = np.arange(0., 1., 0.01)
count = 0
for i in x:
    pop_output.append(p.Population(1, p.IF_curr_exp, cell_params_lif))
    poisson_spikes = mu.poisson_generator(i*max_rate, 0, runtime)
    pop_source.append( p.Population(1, p.SpikeSourceArray, {'spike_times' : poisson_spikes}) )
    p.Projection(pop_source[count], pop_output[count], ee_connector, target='excitatory')
    pop_output[count].record()
    count += 1
count = 0
for i in x:
    cell_params_lif['i_offset'] = i
    pop_list.append(p.Population(1, p.IF_curr_exp, cell_params_lif))
    pop_list[count].record()
    count += 1
pop_list[count-1].record_v()
p.run(runtime)

rate_I = np.zeros(count)
rate_P = np.zeros(count)
for i in range(count):
    spikes = pop_list[i].getSpikes(compatible_output=True)
    rate_I[i] = len(spikes)/run_s
    spikes = pop_output[i].getSpikes(compatible_output=True)
    rate_P[i] = len(spikes)/run_s
#plot_spikes(spikes, 'Current = 10. mA')
plt.plot(x, rate_I, label='current',)
plt.plot(x, rate_P, label='Poisson input')
plt.plot(x, (x-0.1)*126./0.7, label='linear')
plt.legend(loc='upper left', shadow=True)
#plt.show()
plt.draw()
plt.savefig('ReLU.pdf')
plt.clf()
v = pop_list[count-1].get_v(compatible_output=True)

if v is not None:
    ticks = len(v) / 1
    plt.figure()
    plt.xlabel('Time/ms')
    plt.ylabel('v')
    plt.title('v')
    for pos in range(0, 1, 20):
        v_for_neuron = v[pos * ticks: (pos + 1) * ticks]
        plt.plot([i[2] for i in v_for_neuron])
    plt.show()

p.end()







