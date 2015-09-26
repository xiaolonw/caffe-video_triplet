#!/usr/bin/env sh

TOOLS=/nfs/hn46/xiaolonw/video_cnncode2/caffe-video_triplet/build/tools

$TOOLS/caffe train --solver=solver.prototxt --weights=/home/xiaolonw/video_models/unsup_vgg/video__iter_62809.caffemodel




# /nfs/hn46/xiaolonw/video_cnncode/caffe-video_rank/scripts_video2/rank_gray_8m_1fc_pre2/imagenet_solver.prototxt /nfs/hn38/users/xiaolonw/video_models3/rank_gray_8m_1fc_pre2/bef_train_again/video__iter_435000


