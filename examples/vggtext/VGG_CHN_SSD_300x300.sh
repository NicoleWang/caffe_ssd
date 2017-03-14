cd /home/wangyuzhuo/Experiments/ssd_git/caffe_ssd
./build/tools/caffe train \
--solver="models/vggtext/SSD_300x300/solver.prototxt" \
--weights="models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel" \
--gpu 2,3 2>&1 | tee logs/vggtext/SSD_300x300/VGG_CHN_SSD_300x300.log
