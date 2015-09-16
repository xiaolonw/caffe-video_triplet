#!/usr/bin/env sh                                                                                                i

# test_net_seg.bin test_proto pre_train_model label.txt outputfolder [CPU/GPU]

ROOTFILE=/nfs/hn46/xiaolonw/video_cnncode/caffe-video_rank

GLOG_logtostderr=1 $ROOTFILE/build_compute-0-5/tools/showParameters.bin $ROOTFILE/scripts_video/rank_30f_hard_full/imagenet_train.prototxt  /nfs/hn38/users/xiaolonw/video_models/rank_30f_hard_monerelu_gray/video__iter_160000

