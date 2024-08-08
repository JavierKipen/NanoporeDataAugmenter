import tensorflow as tf
import numpy as np
import time
import pdb

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')

import matplotlib.pyplot as plt

length=700; #Length of the used dataset
n_evs=1000;
ev_data_array=np.tile(np.arange(length), (n_evs, 1)) #No transformation
plt.plot(ev_data_array[0,:])

import time
noise=np.random.randn(n_evs,length)
print("Loading lib")
browAug=tf.load_op_library('./../ext/BrownianDataAugmentation/TFBrowAug/browAug.so')
print("Computing output")
start=time.time()

data_out,ev_len_out=browAug.BrowAug(data_in=ev_data_array,noise=noise)
end=time.time()
print("Total Run time: " + str(end-start))

data_out=data_out.numpy();
ev_len_out=ev_len_out.numpy();
data_out=data_out.reshape((-1,length))
np.shape(data_out)

plt.plot(ev_data_array[0,:])
for i in range(100):
    plt.plot(data_out[i,:])
plt.show()