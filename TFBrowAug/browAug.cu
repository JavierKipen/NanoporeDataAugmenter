#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <time.h>

#include "browAug.h"


#define TPB 64


#define N_POINTS_PER_EV 700

__device__ unsigned int generate_new_xs(float * new_xs,float * noise,const unsigned long ev_start);
__device__ void interp_linear(float * x_out,float * y_in,float * y_out, const unsigned int length_x);
__global__ void brow_Kernel(float * data_in, float * data_out,float * noise,float *ev_len_out,unsigned int nEvents);


__global__ void copyKernel(float * data_in, float * data_out,unsigned int len)
{
    const int evIdx = blockIdx.x*TPB+ threadIdx.x;
    if(evIdx<len)
    {
        for(unsigned int i = 0; i < len; i++)
            data_out[i]=data_in[i];
    }
}

__device__ unsigned int generate_new_xs(float * new_xs,float * noise,const unsigned long ev_start)
{ 
    unsigned int array_index=0,final_len;
    float curr_x_index=0;
    float x_to_interpol[N_POINTS_PER_EV]; //Obtains the new sampling points for the output (xs where it will sample) 

    for(unsigned int new_idx=0; new_idx < N_POINTS_PER_EV; new_idx++)
    {
        x_to_interpol[new_idx]=curr_x_index;
        curr_x_index += (1+noise[ev_start+new_idx]);
        if (curr_x_index < 0) //We dont allow negative indexes
            curr_x_index=0;
        if (curr_x_index > N_POINTS_PER_EV) //If the next index goes beyond the maximum, it finishes 
            break;
    }
    //array_index--;
    final_len=new_idx;
    return final_len; //new length of the event
}

__device__ void interp_linear(float * x_out,float * y_in,float * y_out, const unsigned int length_x)
{
    float p_y_i,n_y_i;
    unsigned int p_x_i,n_x_i;
    for(unsigned int i=0;i<N_POINTS_PER_EV;i++) //Every point in y_out.
    {
        if (i < length_x)
        {
            p_x_i=int(x_out[i]);n_x_i=p_x_i+1; //We find the integers that are previous and next to our xi, we will interpolated with each.
            p_y_i=y_in[p_x_i];n_y_i=y_in[n_x_i]; //Then we have the values of y_in evaluated at those integers
            y_out[i] = p_y_i + ((n_y_i-p_y_i)/(n_x_i-p_x_i)) * (x_out[i] - p_x_i); //Linear interpolation between them.

        }
        else
            y_out[i]=std::nan; //When the output event is shorter than 700, we fill with nan.
    }
}

__global__ void brow_Kernel(float * data_in, float * data_out,float * noise,float *ev_len_out,unsigned int nEvents)
{
    const int evIdx = blockIdx.x*TPB+ threadIdx.x;
    const unsigned long ev_start=ev_Idx*N_POINTS_PER_EV;
    float shared_ev_data[N_POINTS_PER_EV];float new_xs[N_POINTS_PER_EV];
    unsigned int new_length;
    if (evIdx <nEvents) //checking boundaries
    {
        for(unsigned int i=0;i<N_POINTS_PER_EV;i++) //Retrieve event to local array
            shared_ev_data[i]=data_in[ev_start+i];
        new_length=generate_new_xs(new_xs,noise,ev_start);
        ev_len_out[evIdx]=(float)new_length;
        interp_linear(new_xs,shared_ev_data,&(data_out[ev_start]),new_length);
    }
}

void BrowLauncher(cudaStream_t& stream,float * data_in, float * data_out,float * noise,float *ev_len_out,unsigned int nEvents) {
  brow_Kernel<<<nEvents/TPB + 1, TPB,0,stream>>>(data_in,data_out,noise,ev_len_out,nEvents);
  cudaDeviceSynchronize();
}
