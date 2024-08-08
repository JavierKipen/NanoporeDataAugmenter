import cupy as cp
import numpy as np


class DataAugmenterV2():
    def __init__(self,brow_std=0.8,magnitude_rel_std=0.05,stretch_rel_std=0.06,noise_std=0.02,
                 fill_noise_std=0.008,apply_brownian_aug=True,gpu_computation=True):
        ##Saves parameters for the augmentations
        self.brow_std=brow_std;
        self.magnitude_rel_std=magnitude_rel_std; 
        self.stretch_rel_std=stretch_rel_std;
        self.noise_std=noise_std;
        self.gpu_computation=gpu_computation;
        self.apply_brownian_aug=apply_brownian_aug;
        self.fill_noise_std=fill_noise_std;
    
    def augment(self,X):
        if self.gpu_computation:
            return self.augment_gpu(X);
        else:
            return self.augment_cpu(X);

# ########### GPU implementation with cupy ############

    def augment_gpu(self,X):
        
        ##Applies all the augmentations to the input data
        n_samples=len(X);sample_len=len(X[0]); #Gets the dimensions of the input data
        if self.apply_brownian_aug: ##Brownian augmentation
            incs=cp.ones(X.shape) ##To increment one sample for each new index
            incs[:,0]=0;
            interp_ids=cp.cumsum(incs+cp.random.normal(0, self.brow_std, X.shape),axis=1) #Formula from paper
            interp_ids[interp_ids<0]=0; #Sets the negative values to 0
        else:
            interp_ids=cp.tile(cp.reshape(cp.arange(sample_len),(1,sample_len)),(n_samples,1)) 
        stretch_factor=cp.random.normal(1,self.stretch_rel_std,size=(n_samples,)) # stretch factors which multiply each row of indexes.
        interp_ids_stretched=(interp_ids.T * stretch_factor).T#https://stackoverflow.com/questions/18522216/multiplying-across-in-a-numpy-array
        x_aug = self.linear_interpolate_and_fill_gpu(X,interp_ids_stretched);
        x_aug = x_aug * cp.abs(cp.random.normal(1, self.magnitude_rel_std, n_samples).reshape((-1,1)) )
        x_aug = x_aug + cp.random.normal(0, self.noise_std, x_aug.shape)
        
        return x_aug;


    def linear_interpolate_and_fill_gpu(self,X,interp_ids): ##Does cp.interp and fills with noise(when needed) with all the traces, in a way that it doesnt need a for loop (optimal for GPU) 
        ##X: 2D array with the base traces
        ##interp_ids: 2D array with the indices (within the same trace) to interpolate
        n_samples=len(X);sample_len=len(X[0]); #Gets the dimensions of the input data
        
        samples_to_exclude_mask=interp_ids>(sample_len-1); ##Samples that have x that we dont have are excluded
        interp_ids[samples_to_exclude_mask]=0; #Interpolating indexes above 699 are set to 0
        min_len=cp.min(cp.argwhere(samples_to_exclude_mask)[0,:]) #Minimum length of the traces after the interpolation, used for the noise filling
        fill_noise=cp.random.normal(0,self.fill_noise_std,size=(int(n_samples),int(sample_len-min_len))); #Noise to fill the traces that are shorter than the rest after the interpolation

        interp_ids_floor=cp.floor(interp_ids).astype(int); #Floor of interpolating indexes

        row_indicator=cp.tile(cp.reshape(cp.arange(n_samples),(n_samples,1)),(1,sample_len)) #Matrix where each element is the $    x_base=xs[row_indicator,interpol_idxs_floor]; #using the floor of the interpolating indexes to get the the value for th$
        x_base=X[row_indicator,interp_ids_floor]; #using the floor of the interpolating indexes to get the the value for th$

        multiplier= interp_ids-interp_ids_floor; #Interpolating values between 0 and 1 (n_samples,1)
        difference=X[row_indicator,interp_ids_floor+1]-x_base; #Difference between the Y(x) where x is the floor and floor $

        x_interpolated = x_base + cp.multiply(multiplier, difference) #Linear difference.

        #x_interpolated_sub_ending=x_interpolated[:,min_len:];
        x_interpolated[samples_to_exclude_mask]=fill_noise[samples_to_exclude_mask[:,min_len:]];
        return x_interpolated


############ CPU implementation with numpy ############
    def augment_cpu(self,X):
        ##Applies all the augmentations to the input data
        n_samples=len(X);sample_len=len(X[0]); #Gets the dimensions of the input data
        if self.apply_brownian_aug: ##Brownian augmentation
            incs=np.ones(X.shape) ##To increment one sample for each new index
            incs[:,0]=0;
            interp_ids=np.cumsum(incs+np.random.normal(0, self.brow_std, X.shape),axis=1) #Formula from paper
            interp_ids[interp_ids<0]=0; #Sets the negative values to 0
        else:
            interp_ids=np.tile(np.reshape(np.arange(sample_len),(1,sample_len)),(n_samples,1)) 
        stretch_factor=np.random.normal(1,self.stretch_rel_std,size=(n_samples,)) # stretch factors which multiply each row of indexes.
        interp_ids_stretched=(interp_ids.T * stretch_factor).T#https://stackoverflow.com/questions/18522216/multiplying-across-in-a-numpy-array
        x_aug = self.linear_interpolate_and_fill_gpu(X,interp_ids_stretched);
        x_aug = x_aug * np.abs(np.random.normal(1, self.magnitude_rel_std, n_samples).reshape((-1,1)) )
        x_aug = x_aug + np.random.normal(0, self.noise_std, x_aug.shape)
        
        return x_aug;

    def linear_interpolate_and_fill_cpu(self,X,interp_ids): ##Does cp.interp and fills with noise(when needed) with all the traces, in a way that it doesnt need a for loop (optimal for GPU) 
        ##X: 2D array with the base traces
        ##interp_ids: 2D array with the indices (within the same trace) to interpolate
        n_samples=len(X);sample_len=len(X[0]); #Gets the dimensions of the input data
        
        samples_to_exclude_mask=interp_ids>(sample_len-1); ##Samples that have x that we dont have are excluded
        interp_ids[samples_to_exclude_mask]=0; #Interpolating indexes above 699 are set to 0
        min_len=np.min(np.argwhere(samples_to_exclude_mask)[0,:]) #Minimum length of the traces after the interpolation, used for the noise filling
        fill_noise=np.random.normal(0,self.fill_noise_std,size=(n_samples,sample_len-min_len)); #Noise to fill the traces that are shorter than the rest after the interpolation

        interp_ids_floor=np.floor(interp_ids).astype(int); #Floor of interpolating indexes

        row_indicator=np.tile(np.reshape(np.arange(n_samples),(n_samples,1)),(1,sample_len)) #Matrix where each element is the $    x_base=xs[row_indicator,interpol_idxs_floor]; #using the floor of the interpolating indexes to get the the value for th$
        x_base=X[row_indicator,interp_ids_floor]; #using the floor of the interpolating indexes to get the the value for th$

        multiplier= interp_ids-interp_ids_floor; #Interpolating values between 0 and 1 (n_samples,1)
        difference=X[row_indicator,interp_ids_floor+1]-x_base; #Difference between the Y(x) where x is the floor and floor $

        x_interpolated = x_base + np.multiply(multiplier, difference) #Linear difference.

        #x_interpolated_sub_ending=x_interpolated[:,min_len:];
        x_interpolated[samples_to_exclude_mask]=fill_noise[samples_to_exclude_mask[:,min_len:]];
        return x_interpolated

############ CPU slow implementation for control and test ############
    def linear_interpolate_and_fill_cpu(self,X,interp_ids): ##Does cp.interp and fills with noise(when needed) with all the traces, in a way that it doesnt need a for loop (optimal for GPU) 
        ##X: 2D array with the base traces
        ##interp_ids: 2D array with the indices (within the same trace) to interpolate
        n_samples=len(X);sample_len=len(X[0]); #Gets the dimensions of the input data
        X_out=np.zeros((n_samples,sample_len));
        for i in range(n_samples):   
            x_resampled=np.interp(interp_ids[i,interp_ids[i,:]<sample_len],np.arange(sample_len),X[i])
            X_out[i,:len(x_resampled)]=x_resampled;
            if len(x_resampled)<sample_len:
                X_out[i,len(x_resampled):]=np.random.normal(0,self.fill_noise_std,sample_len-len(x_resampled));
    
        return X_out

############ Tests to validate implementation ############
def test_numerical_augmentation(X,use_brow_aug=True):
    ##Tests the augmentations (TBD)
    return 0;

def show_few_examples_augmentation(da,X=None,n_show=10,ex_translocation_type="arange"):
    ##Tests the augmentations
    import matplotlib.pyplot as plt
    if X is None:
        evs_size=700;
        example_translocation=np.zeros((700,));
        ev_len=300;
        example_translocation[50:350]=-0.95-0.1*np.arange(ev_len)/ev_len;
        example_translocation[150:175]-=1;
        example_translocation= np.arange(evs_size) if ex_translocation_type=="arange" else example_translocation;
        n_evs=300;
        X=np.tile(example_translocation, (n_evs, 1))
    
    x_aug=da.augment(X)

    for i in range(n_show):
        plt.plot(x_aug[i,:]) 
        
    plt.show()

if __name__ == "__main__":
    
    gpu_computation=True;
    print("Only Brownian");
    da = DataAugmenterV2(brow_std=0.9,magnitude_rel_std=0.0001,stretch_rel_std=0.0001,noise_std=0.0001,fill_noise_std=0.0001,apply_brownian_aug=True,gpu_computation=gpu_computation)
    show_few_examples_augmentation(da)
    print("Only stretch");
    da = DataAugmenterV2(brow_std=0.9,magnitude_rel_std=0.0001,stretch_rel_std=0.08,noise_std=0.0001,fill_noise_std=0.0001,apply_brownian_aug=False,gpu_computation=gpu_computation)
    show_few_examples_augmentation(da)
    print("Only Magnitude");
    da = DataAugmenterV2(brow_std=0.9,magnitude_rel_std=0.05,stretch_rel_std=0.0001,noise_std=0.0001,fill_noise_std=0.0001,apply_brownian_aug=False,gpu_computation=gpu_computation)
    show_few_examples_augmentation(da)
    print("Only Noise");
    da = DataAugmenterV2(brow_std=0.9,magnitude_rel_std=0.05,stretch_rel_std=0.0001,noise_std=0.0001,fill_noise_std=0.0001,apply_brownian_aug=False,gpu_computation=gpu_computation)
    show_few_examples_augmentation(da)
    print("All together");
    da = DataAugmenterV2(brow_std=0.9,magnitude_rel_std=0.05,stretch_rel_std=0.03,noise_std=0.05,fill_noise_std=0.008,apply_brownian_aug=True,gpu_computation=gpu_computation)
    show_few_examples_augmentation(da)
