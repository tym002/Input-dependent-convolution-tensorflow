# Input-dependent-convolution-tensorflow
A tensorflow based custom convolutional layers. The convolutional weights depend on the input. 

In regular convolution, the convolutional kernels are independtly learned. All examples in a mini_batch have the same convolutional kernels. 

In this implementation, the convolutional kernels are conditioned on the input features. As a results, the convolutional kernels are different for different inputs within a mini_batch. 
