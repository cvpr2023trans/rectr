# This scripts check if CUDA is available, and prints out all GPU details

import torch.cuda as cuda

if cuda.is_available():
    print("CUDA is available")
    print("CUDA device count: {}".format(cuda.device_count()))
    
    for i in range(cuda.device_count()):
        print("CUDA device {}: {}".format(i, cuda.get_device_name(i)))

        # Print properties and characteristics of the GPU
        print("CUDA device {}: {}".format(i, cuda.get_device_properties(i)))
        print("CUDA device {}: {}".format(i, cuda.get_device_capability(i)))

else:
    print("CUDA is not available")