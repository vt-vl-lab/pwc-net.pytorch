## PWC-Net (PyTorch v1.0.1)

Pytorch implementation of [PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume](https://arxiv.org/pdf/1709.02371.pdf). We made it as a off-the-shelf package:
- After installation, just copy the whole folder `PWC_src` to your codebase to use. See demo.py for details.

### Environment

This code has been test with Python3.6 and PyTorch1.0.1, with a Tesla K80 GPU. The system is Ubuntu 14.04, and the CUDA version is 10.0. All the required python packages can be found in `requirements.txt`.

### Installation 

    # install custom layers
    cd PWC_src/correlation_package
    python setup.py install

Note: you might need to add `gencode` [here](https://github.com/vt-vl-lab/pwc-net.pytorch/blob/master/PWC_src/correlation_package/setup.py#L9), according to the GPU you use. You can find more information about `gencode` [here](https://developer.nvidia.com/cuda-gpus) and [here](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).

### Converted Caffe Pre-trained Models
You can find them in `models` folder.

### Inference mode
Modify the path to your input, then

```
python demo.py
```    

If installation is sucessful, you should see the following:
![FlowNet2 Sample Prediction](https://github.com/vt-vl-lab/pwc-net.pytorch/blob/master/misc/demo.png)
   
### Reference 
If you find this implementation useful in your work, please acknowledge it appropriately and cite the paper using:
````
@inproceedings{sun2018pwc,
  title={PWC-Net: CNNs for optical flow using pyramid, warping, and cost volume},
  author={Sun, Deqing and Yang, Xiaodong and Liu, Ming-Yu and Kautz, Jan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={8934--8943},
  year={2018}
}
````

### Acknowledgments
* [sniklaus/pytorch-pwc](https://github.com/sniklaus/pytorch-pwc): Network defintion and converted PyTorch model weights.
* [NVIDIA/flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch): Correlation module.
