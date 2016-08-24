
# caffe-video_triplet

This code is developed based on Caffe: [project site](http://caffe.berkeleyvision.org).

This code is the implementation for training the siamese-triplet network in the paper:

**Xiaolong Wang** and Abhinav Gupta. Unsupervised Learning of Visual Representations using Videos. Proc. of IEEE International Conference on Computer Vision (ICCV), 2015. [pdf](http://www.cs.cmu.edu/~xiaolonw/papers/unsupervised_video.pdf)

Codes
----

Training scripts are in rank_scripts/rank_alexnet: 

For implementation, since the siamese networks share the weights, so there is only one network in prototxt. 

The input of the network is pairs of image patches. For each pair of patches, they are taken as the similar patches in the same video track. We use the label to specify whether the patches come from the same video, if they come from different videos they will have different labels (it does not matter what is the number, just need to be integer). In this way, we can get the third negative patch from other pairs with different labels. 

In the loss, for each pair of patches, it will try to find the third negative patch in the same batch. There are two ways to do it, one is random selection, the other is hard negative mining. 

In the prototxt: 

```txt
layer {		
	name: "loss"	
	type: "RankHardLoss" 	
	rank_param{		
		neg_num: 4	
		pair_size: 2 	
		hard_ratio: 0.5 	
		rand_ratio: 0.5 	
		margin: 1 	
	} 	
	bottom: "norml2" 	
	bottom: "label" 	
}
```

neg_num means how many negative patches you want for each pair of patches, if it is 4, that means there are 4 triplets. pair_size = 2 just means inputs are pairs of patches. hard_ratio = 0.5 means half of the negative patches are hard examples, rand_ratio = 0.5 means half of the negative patches are randomly selected. For start, you can just set rand_ratio = 1 and hard_ratio = 0. The margin for contrastive loss needs to be designed for different tasks, trying to set  margin = 0.5 or 0.1 might make a difference for other tasks. 


Models
----

We offer two models trained with our method: 

[color model](http://ladoga.graphics.cs.cmu.edu/xiaolonw/unsup_models/color_model.caffemodel) is trained with RGB images. 	
[gray model](http://ladoga.graphics.cs.cmu.edu/xiaolonw/unsup_models/gray_model.caffemodel) is trained with gray images (3-channel inputs). 	
[prototxt](https://github.com/xiaolonw/caffe-video_triplet/blob/master/rank_scripts/rank_alexnet/unsup_net_deploy.prototxt) is the prototxt for both models.	
[mean](https://github.com/xiaolonw/caffe-video_triplet/blob/master/rank_scripts/rank_alexnet/video_mean.binaryproto) is the mean file.	

In case our server is down, the models can be downloaded from dropbox:
[color model](https://dl.dropboxusercontent.com/u/334666754/iccv2015/color_model.caffemodel) is trained with RGB images. 	
[gray model](https://dl.dropboxusercontent.com/u/334666754/iccv2015/gray_model.caffemodel) is trained with gray images (3-channel inputs). 

Training Patches
----
The unsupervised mined patches can be downloaded from here:
https://www.dropbox.com/sh/b9rgd8nkh498kuy/AACt6Gk8V7-f8Yq7qVSCs5TGa?dl=0

Each tar file contains different patches.

The example of the training list can be downloaded from here:
https://dl.dropboxusercontent.com/u/334666754/unsup_patches/trainlist.txt









