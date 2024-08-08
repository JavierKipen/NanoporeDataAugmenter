
#pragma once

#include "cuda.h"
#include <stdint.h>

void BrowLauncher(cudaStream_t& stream,float * data_in, float * data_out,float * noise,float *ev_len_out,unsigned int nEvents);
