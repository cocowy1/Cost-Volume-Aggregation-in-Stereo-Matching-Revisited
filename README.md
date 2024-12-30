# DCANet
Cost aggregation plays a critical role in existing stereo matching methods. In this paper, we revisit cost aggregation in stereo matching from disparity classification and propose
a generic yet efficient Disparity Context Aggregation (DCA) module to improve the performance of CNN-based methods. Our approach is based on an insight that a coarse disparity
class prior is beneficial to disparity regression. To obtain such a prior, we first classify pixels in an image into several disparity classes and treat pixels within the same class as homogeneous regions. We then generate homogeneous region representations and incorporate these representations into the cost volume to suppress irrelevant information while enhancing the matching
ability for cost aggregation. With the help of homogeneous region representations, efficient and informative cost aggregation can be achieved with only a shallow 3D CNN. Our DCA module is fully differentiable and well-compatible with different network architectures, which can be seamlessly plugged into existing networks to improve performance with small additional
overheads. It is demonstrated that our DCA module can effectively exploit disparity class priors to improve the performance of cost aggregation. Based on our DCA, we design a highly accurate network named DCANet, which achieves state-of-the-art performance on several benchmarks.


# DCA (Disparity Context Aggregation) Module Overview
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

# Disparity Classficiation Visualization on KITTI
<img width="900" src="https://github.com/cocowy1/DCANet/blob/main/figs/kitti_mask.png"/></div>

#  Visualization of Grad-CAM
<img width="900" src="https://github.com/cocowy1/DCANet/blob/main/figs/heatmap%20comparison.png"/></div>


# 1. Train on SceneFlow
Run `main.py` to train on the SceneFlow dataset. Please update datapath in `main.py` as your training data path.

# 2. Train on KITTI \& ETH3D 
Run `train_kitti.py` or `train_eth3d` to finetune on the different real-world datasets, such as KITTI 2012, KITTI 2015, and ETH3D.

To generate prediction results on the test set of the KITTI dataset, you can run `evaluate_kitti.py`. 
The inference time can be printed  once you run `evaluate_kitti.py`. 
And the inference results on the KITTI dataset can be directly submitted to the online evaluation server for benchmarking.

# 3. Inference 
Run `my_img.py` to finetune on the  KITTI 2012, KITTI 2015. Please update datapath in `my_img.py` as your testing data path.


# Acknowledgements

This project is based on [GwcNet](https://github.com/xy-guo/GwcNet). We thank the original authors for their excellent works.
