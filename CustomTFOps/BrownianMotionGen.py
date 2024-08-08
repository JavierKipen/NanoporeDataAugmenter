# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 14:44:52 2022

@author: JK-WORK
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import tensorflow as tf
import time
import copy
import ipdb
import pathlib


        
def transform_event_yaxis_base_vectorized(event_data_array, proc_std, spline_inter_fact=3, same_len=True, fill_noise_std=None):
    n_evs=np.shape(event_data_array)[0];length=np.shape(event_data_array)[1];
    transformed_array=np.zeros((n_evs,length))
    idxs_per_ev=np.tile(np.arange(length), (n_evs, 1)) #No transformation
    idxs_per_ev=idxs_per_ev.astype("float")
    idxs_per_ev[:,1:] += np.cumsum(proc_std*np.random.randn(n_evs,length-1),axis=1); #Adds brownian motion
    idxs_per_ev[idxs_per_ev<0]=0; #Solve negative problem
    
    
    if not(spline_inter_fact is None):
        x_for_cubic = np.linspace(0, length-1, num=spline_inter_fact*length)
        f = interpolate.interp1d(np.arange(length), event_data_array, kind='cubic')
        interpol=f(x_for_cubic);
    else:
        x_interpol=np.arange(length);
    for ev_idx in range(n_evs): 
        new_x_idxs=idxs_per_ev[ev_idx,:];
        if np.any(new_x_idxs>length):
            new_x_idxs=new_x_idxs[:(np.argwhere(new_x_idxs>length)[0][0]+1)] #Keeps values within the translocation
        if not(spline_inter_fact is None):
            data_transformed_aux=np.interp(new_x_idxs,x_for_cubic,interpol[ev_idx,:])
        else:
            data_transformed_aux=np.interp(new_x_idxs,x_interpol,event_data_array[ev_idx,:])
        if len(data_transformed_aux)< length:
            if fill_noise_std is None:
                data_transformed_aux = np.concatenate( (data_transformed_aux,np.zeros(length-len(data_transformed_aux))))
            else:
                data_transformed_aux = np.concatenate( (data_transformed_aux,fill_noise_std*np.random.randn(length-len(data_transformed_aux))))
        transformed_array[ev_idx,:]=data_transformed_aux;
    return transformed_array


class brow_aug_GPU_opt():
    def __init__(self,proc_std=0.5,event_len_in_array=False):
        self.std=proc_std;
        par_folder=str(pathlib.Path(__file__).parent.resolve());
        self.browAug=tf.load_op_library(par_folder+'/TFBrowAug/browAug.so')
        ipdb.set_trace();
        self.event_len_in_array=event_len_in_array;
    def aug(self,data_in,noise=None):
        if noise is None:
            noise=self.std*np.random.randn(np.shape(data_in)[0],2*np.shape(data_in)[1]); 
        if self.event_len_in_array:
            #ipdb.set_trace();
            data_out,ev_len_out=self.browAug.BrowAug(data_in=data_in[:,:-1,0],noise=noise)
            data_out=data_out.numpy();
            ev_len_out=ev_len_out.numpy()
            data_out=data_out.reshape((-1,np.shape(data_in)[1]-1))
            # data_out=data_in[:,:-1,0]
            # ev_len_out=data_in[:,-1,0]
            ret=np.zeros(np.shape(data_in))
            ret[:,:-1,0]=data_out
            ret[:,-1,0]=ev_len_out
            return ret;
        else:
            data_out,ev_len_out=self.browAug.BrowAug(data_in=data_in,noise=noise)
            data_out=data_out.numpy();
            data_out=data_out.reshape((-1,np.shape(data_in)[1]))
            ev_len_out=ev_len_out.numpy()
            return data_out,ev_len_out;

### Functions that mimic GPU functions, to test that they work as expected

def gen_new_xs(ev_len,noise_array,nfinal=500,interp="Linear"):
    curr_x_idx=0;x_s=[];array_idx=0;
    while curr_x_idx < ev_len - 1:
        x_s.append(curr_x_idx)
        curr_x_idx += 1 + noise_array[array_idx]
        curr_x_idx= 0 if curr_x_idx < 0 else curr_x_idx;
        array_idx+=1;
    if interp=="Linear":
        new_xs=np.interp(np.linspace(0,len(x_s)-1,nfinal), np.arange(len(x_s)), x_s)
    else:
        f = interpolate.interp1d(np.arange(len(x_s)), x_s,kind="cubic")
        new_xs=f(np.linspace(0,len(x_s)-1,nfinal))
    return new_xs,array_idx;
def brow_aug_lin_interp_as_GPU(data_in,noise,interp="Linear"):
    nfinal=np.shape(data_in)[1]
    n_events=np.shape(data_in)[0]
    out=np.zeros((n_events,nfinal))
    out_lens=[];
    for i in range(n_events):
        curr_data=data_in[i,:];
        ev_len=np.argwhere(np.isnan(curr_data))[0][0] if np.argwhere(np.isnan(curr_data)).size>0 else len(curr_data);
        new_xs,new_len=gen_new_xs(ev_len,noise[i,:],nfinal=nfinal,interp=interp)
        if interp=="Linear":
            out[i,:]=np.interp(new_xs, np.arange(ev_len), curr_data[:ev_len])
        else:
            f = interpolate.interp1d(np.arange(ev_len), curr_data[:ev_len],kind="cubic")
            out[i,:]=f(new_xs)
        out_lens.append(new_len)
    return out,out_lens


### Tests that we have 

def test_n_profile_vectorized_brownian():
    length=700; #Length of the used dataset
    n_evs=1000;
    ev_data_array=np.tile(np.arange(length), (n_evs, 1)) #No transformation
    start = time.time()
    ev_data_array_transf=transform_event_yaxis_base_vectorized(ev_data_array, 1, fill_noise_std=0.1);
    end = time.time()
    print(end - start)

def test_GPU_brownian():
    length=500; #Length of the used dataset
    n_evs=1000;
    ev_data_array=np.tile(np.arange(length), (n_evs, 1)) #No transformation
    browAug=brow_aug_GPU_opt(proc_std=0.5);
    print("Results:")
    #print("Some ev lengths: " + str(ev_len_out[:10]))
    
    
    
    data_out,ev_len_out=browAug.aug(ev_data_array)
    print("First array out:")
    print("["+ str(data_out[0,:3]) + "... "+str(data_out[0,-2:]) +"]")
    print("Last array out:")
    print("["+ str(data_out[-1,:3]) + "... "+str(data_out[-1,-2:]) +"]")
    
def test_GPU_brownian_validation(w_GPU=False):
    len_out=700;n_ev=3; 
    evs=np.zeros((n_ev,len_out)); std=1;
    evs[:]=np.nan;
    evs[0,:350]=np.cos(0.1*np.arange(350)); #Try one normal input
    evs[1,:450]=np.cos(0.05*np.arange(450)); #Try one normal input
    evs[2,:]=np.cos(0.15*np.arange(len_out)); #Try one max size input
    np.random.seed(1);
    noise=std*np.random.randn(n_ev,2*len_out); 
    if w_GPU:
        browAug=brow_aug_GPU_opt();
        evs=evs.reshape(n_ev,len_out,1)
        out_GPU,out_lens_GPU=browAug.aug(evs,noise=noise);
        np.save("OutGPU.npy",out_GPU);np.save("OutLensGPU.npy",out_lens_GPU);np.save("Noise.npy",noise)
    else:
        out_GPU=np.load("OutGPU.npy");out_lens_GPU=np.load("OutLensGPU.npy");noise=np.load("Noise.npy")
    out,out_lens = brow_aug_lin_interp_as_GPU(evs,noise)
    outCubic,outCubic_lens = brow_aug_lin_interp_as_GPU(evs,noise, interp="Cubic")
    plt.figure()
    plt.plot(out[1,:],label="CPU Linear interp trace")
    plt.plot(outCubic[1,:],label="CPU Cubic interp trace")
    plt.plot(out_GPU[1,:],label="GPU Linear interp trace")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #test_n_profile_vectorized_brownian();
    test_GPU_brownian_validation(True)
