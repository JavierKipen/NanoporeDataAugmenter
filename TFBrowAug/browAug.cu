#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <time.h>

#include "browAug.h"


#define TPB 64


#define N_POINTS_PER_EV 512
#define N_NOISE_POINTS_PER_SAMPLE 2

__device__ unsigned int retrieveEvent(const float * data_in,float * shared_ev_data,const unsigned int ev_Idx);
__device__ unsigned int generate_new_xs(float * new_xs,float * noise,const int evIdx,const unsigned int ev_len);
__device__ void interp_linear(float * x_out,float * y_b,float * y_out, const unsigned int length_out, const unsigned int length_b);
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
__device__ unsigned int retrieveEvent(const float * data_in,float * shared_ev_data,const unsigned int ev_Idx)
{
    const unsigned long ev_start=ev_Idx*N_POINTS_PER_EV;
    unsigned int ev_len=0;
    for(unsigned int i=0;i<N_POINTS_PER_EV;i++)
    {
        shared_ev_data[i]=data_in[ev_start+i];
        if (isnan(shared_ev_data[i]))
        { ev_len=i;break;}
    }
    if (ev_len==0)
        ev_len=N_POINTS_PER_EV; //In case it didnt break the for_loop
    return ev_len;
}

__device__ unsigned int generate_new_xs(float * new_xs,float * noise,const int evIdx,const unsigned int ev_len)
{ 
    unsigned int array_index=0,final_len;
    float curr_x_index=0;
    float temp_array[2*N_POINTS_PER_EV]; //Brownian transform of the event 
    float x_to_interpol[2*N_POINTS_PER_EV]; //Brownian transform of the event 

    while(curr_x_index < ev_len-1)
    {
        temp_array[array_index]=curr_x_index;
        curr_x_index+= (1+noise[evIdx*N_POINTS_PER_EV*N_NOISE_POINTS_PER_SAMPLE+array_index]);
        if (curr_x_index < 0)
            curr_x_index=0;
        array_index++;
    }
    //array_index--;
    final_len=array_index;
    float step=(((float)final_len-1)/((float)N_POINTS_PER_EV-1));
    for(unsigned int i=0;i<N_POINTS_PER_EV;i++)
        x_to_interpol[i]=step*i;
    interp_linear(x_to_interpol,temp_array,new_xs, N_POINTS_PER_EV, array_index);
    return final_len; //new length of the event
}

__device__ void interp_linear(float * x_out,float * y_b,float * y_out, const unsigned int length_out, const unsigned int length_b)
{
    float p_y_i,n_y_i;
    unsigned int p_x_i,n_x_i;
    for(unsigned int i=0;i<length_out;i++)
    {
        p_x_i=int(x_out[i]);n_x_i=p_x_i+1;
        if (n_x_i >= length_b)
            y_out[i] = y_b[p_x_i];
        else
        {
            p_y_i=y_b[p_x_i];n_y_i=y_b[n_x_i];
            y_out[i] = p_y_i + ((n_y_i-p_y_i)/(n_x_i-p_x_i)) * (x_out[i] - p_x_i);
        }
    }
}

__global__ void brow_Kernel(float * data_in, float * data_out,float * noise,float *ev_len_out,unsigned int nEvents)
{
    const int evIdx = blockIdx.x*TPB+ threadIdx.x;
    float shared_ev_data[N_POINTS_PER_EV];float new_xs[N_POINTS_PER_EV];
    unsigned int ev_len=0;
    const unsigned long ev_start=evIdx*N_POINTS_PER_EV;
    unsigned int new_length;
    if (evIdx <nEvents) //checking boundaries
    {
        ev_len=retrieveEvent(data_in,shared_ev_data,evIdx);
        new_length=generate_new_xs(new_xs,noise,evIdx,ev_len);
        ev_len_out[evIdx]=(float)new_length;
        interp_linear(new_xs,shared_ev_data,&(data_out[ev_start]),N_POINTS_PER_EV, ev_len);
    }
}

void BrowLauncher(cudaStream_t& stream,float * data_in, float * data_out,float * noise,float *ev_len_out,unsigned int nEvents) {
  brow_Kernel<<<nEvents/TPB + 1, TPB,0,stream>>>(data_in,data_out,noise,ev_len_out,nEvents);
  cudaDeviceSynchronize();
}
