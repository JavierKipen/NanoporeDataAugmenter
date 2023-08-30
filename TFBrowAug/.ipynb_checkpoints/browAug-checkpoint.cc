#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

#include "browAug.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;


REGISTER_OP("BrowAug")
    .Input("data_in: float32")
    .Input("noise: float32")
    .Output("data_out: float32")
    .Output("ev_len_out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle nn_prior_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &nn_prior_shape));
        shape_inference::DimensionHandle batch_size = c->Dim(nn_prior_shape, 0);
        //c->set_output(0, c->Vector(batch_size));
        return Status::OK();
    });


class BrowAug : public OpKernel {
    public:
    
    explicit BrowAug(OpKernelConstruction* context) : OpKernel(context) {}


    void Compute(OpKernelContext* context) override {
        //Obtain input tensors
        const Tensor& data_in = context->input(0); 
        const Tensor& noise = context->input(1); 

        TensorShape data_in_shape = data_in.shape();
        TensorShape noiseshape = noise.shape();

        unsigned int n_events = data_in_shape.dim_size(0);
        unsigned int n_points_per_ev = data_in_shape.dim_size(1);
        unsigned int noise_len = noiseshape.dim_size(0);

        // Create output tensor
        TensorShape output_shape;
        output_shape.AddDim(n_events*n_points_per_ev);
        //output_shape.AddDim(n_points_per_ev);
        Tensor* data_out = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &data_out));

        TensorShape output_shape_2;
        output_shape_2.AddDim(n_events);
        Tensor* ev_len_out = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, output_shape_2, &ev_len_out));

        //The next complies, so it will be the base for moving forward
        auto dev = context->eigen_device<Eigen::GpuDevice>();
        auto stream = dev.stream();

        BrowLauncher(stream, (float *) data_in.data(),(float *) data_out->data(),(float *) noise.data(),(float *) ev_len_out->data(),n_events);
    }


};


REGISTER_KERNEL_BUILDER(Name("BrowAug").Device(DEVICE_GPU), BrowAug);
