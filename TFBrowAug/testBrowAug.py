import tensorflow as tf
import numpy as np
import time
import pdb

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')


length=500; #Length of the used dataset
n_evs=100000;
ev_data_array=np.tile(np.arange(length), (n_evs, 1)) #No transformation
noise=np.random.randn(n_evs,2*length)
print("Loading lib")
browAug=tf.load_op_library('./browAug.so')
print("Computing output")
start=time.time()

data_out,ev_len_out=browAug.BrowAug(data_in=ev_data_array,noise=noise)
end=time.time()
print("Total copy time: " + str(end-start))
#pdb.set_trace()
#ev_len_out=ev_len_out.numpy()
data_out=data_out.numpy();
data_out=data_out.reshape((-1,length))
print("Results:")
#print("Some ev lengths: " + str(ev_len_out[:10]))

print("First array out:")
print("["+ str(data_out[0,:3]) + "... "+str(data_out[0,-2:]) +"]")
print("Last array out:")
print("["+ str(data_out[-1,:3]) + "... "+str(data_out[-1,-2:]) +"]")
