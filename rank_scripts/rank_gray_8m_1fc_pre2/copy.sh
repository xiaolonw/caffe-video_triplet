#!/usr/bin/env sh

ROOTFILE=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_joint
MODELDIR=/nfs/hn38/users/xiaolonw/video_models3/rank_gray_8m_1fc_pre2/bef_train_again


GLOG_logtostderr=1 $ROOTFILE/build_compute-0-5/tools/copynet.bin  imagenet_train_rename_2.prototxt $MODELDIR/video__iter_435000 $MODELDIR/video__iter_435000_rename




