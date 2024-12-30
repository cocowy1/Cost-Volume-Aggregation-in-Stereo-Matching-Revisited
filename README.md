# DCANet
Cost aggregation plays a critical role in existing stereo matching methods. In this paper, we revisit cost aggregation in stereo matching from disparity classification and propose
a generic yet efficient Disparity Context Aggregation (DCA) module to improve the performance of CNN-based methods. Our approach is based on an insight that a coarse disparity
class prior is beneficial to disparity regression. To obtain such a prior, we first classify pixels in an image into several disparity classes and treat pixels within the same class as homogeneous regions. We then generate homogeneous region representations and incorporate these representations into the cost volume to suppress irrelevant information while enhancing the matching
ability for cost aggregation. With the help of homogeneous region representations, efficient and informative cost aggregation can be achieved with only a shallow 3D CNN. Our DCA module is fully differentiable and well-compatible with different network architectures, which can be seamlessly plugged into existing networks to improve performance with small additional
overheads. It is demonstrated that our DCA module can effectively exploit disparity class priors to improve the performance of cost aggregation. Based on our DCA, we design a highly accurate network named DCANet, which achieves state-of-the-art performance on several benchmarks.


# Overview
<img width="900" src="https://github.com/cocowy1/DCANet/blob/main/figs/DCA_module.png"/></div>

# Environment
```
Python 3.8
Pytorch 1.6.0
```
# Create a virtual environment and activate it.
```
conda create -n DCANet python=3.8
conda activate DCANet
```


# Dependencies
```
conda install pytorch torchvision torchaudio cudatoolkit=10.3 -c pytorch -c nvidia
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install tqdm
pip install chardet
pip install imageio
pip install thop
pip install timm==0.5.4
```
