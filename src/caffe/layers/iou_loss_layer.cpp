#include <vector>

#include "caffe/layers/iou_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void IouLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                                     const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top); 
}

template <typename Dtype>
void IouLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  int num_boxes = bottom[1]->count() / 4;
  std::vector<int> diff_shape(1,num_boxes);
  diff_.Reshape(diff_shape);
  inter_.Reshape(diff_shape);
  union_.Reshape(diff_shape);
  diff_shape.push_back(4);
  diff_I_.Reshape(diff_shape);
  diff_X_.Reshape(diff_shape);
}

template <typename Dtype>
void IouLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
//  const Dtype margin = Dtype(0.01);
    std::cout << "Into IOU Loss Layer" << std::endl;
  int num_boxes = bottom[1]->count() / 4;
  const Dtype* p_gt_data = bottom[0]->cpu_data();
  const Dtype* p_box_data = bottom[1]->cpu_data();
  Dtype* p_diff = diff_.mutable_cpu_data();
  Dtype* p_inter = inter_.mutable_cpu_data();
  Dtype* p_union = union_.mutable_cpu_data();
  Dtype* p_diffI = diff_I_.mutable_cpu_data();
  Dtype* p_diffX = diff_X_.mutable_cpu_data();
  Dtype loss = Dtype(0);
  
  for (int i = 0; i < num_boxes; ++i){
      Dtype g_x1 = p_gt_data[i * 4];
      Dtype g_y1 = p_gt_data[i * 4 + 1];
      Dtype g_x2 = p_gt_data[i * 4 + 2];
      Dtype g_y2 = p_gt_data[i * 4 + 3];
      Dtype r_x1 = p_box_data[i * 4];
      Dtype r_y1 = p_box_data[i * 4 + 1];
      Dtype r_x2 = p_box_data[i * 4 + 2];
      Dtype r_y2 = p_box_data[i * 4 + 3];
      std::cout << g_x1 << " " << g_y1 << " " << g_x2 <<  " " << g_y2 << std::endl;

      p_diffX[i*4] = r_y1 - r_y2;
      p_diffX[i*4 + 1] = r_x1 - r_x2;
      p_diffX[i*4 + 2] = r_y2 - r_y1;
      p_diffX[i*4 + 3] = r_x2 - r_x1;

      Dtype delta_x = std::min(g_x2, r_x2) - std::max(g_x1, r_x1);
      Dtype delta_y = std::min(g_y2, r_y2) - std::max(g_y1, r_y1);
      if (r_x1 >= g_x1) {
          p_diffI[i*4] = Dtype(-1.0) * delta_y;
      } else {
          p_diffI[i*4] = Dtype(0.0);
      }
      if (r_y1 >= g_y1) {
          p_diffI[i*4 + 1] = Dtype(-1.0) * delta_x;
      } else {
          p_diffI[i*4 + 1] = Dtype(0.0);
      }
      if (r_x2 >= g_x2) {
          p_diffI[i*4 + 2] = Dtype(1.0) * delta_y;
      } else {
          p_diffI[i*4 + 2] = Dtype(0.0);
      } 
      if (r_y2 >= g_y2) {
          p_diffI[i*4 + 3] = Dtype(1.0) * delta_x;
      } else {
          p_diffI[i*4 + 3] = Dtype(0.0);
      }
      std::cout << "diff: " ;
      if (delta_x > Dtype(0) && delta_y > Dtype(0)) {
          Dtype inter_area = delta_x * delta_y;
          Dtype union_area = (g_y2 - g_y1) * (g_x2 - g_x1) + (r_y2 - r_y1) * (r_x2 - r_x1);
          p_inter[i] = inter_area;
          p_union[i] = union_area;
          p_diff[i] = Dtype(-1.0) * std::log( (inter_area / (union_area - inter_area)));
          std::cout << p_diff[i] << " " ;
          loss += p_diff[i];
      }
      std::cout << std::endl;
  }
  top[0]->mutable_cpu_data()[0] = loss / num_boxes;
}

template <typename Dtype>
void IouLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  std::cout << "Into IOU Backword" << std::endl;
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
        const Dtype* p_inter = inter_.cpu_data();
        const Dtype* p_union = union_.cpu_data();
        Dtype* p_diffI = diff_I_.mutable_cpu_data();
        const Dtype* p_diffX = diff_X_.cpu_data();
        for (int j = 0; j < inter_.count(); ++j) {
            Dtype temp = (p_inter[j] + p_union[j]) / (p_inter[j] * p_union[j]);
            for (int k = 0; k < 4; ++k) {
                p_diffI[j*4 + k] = p_diffX[j*4 + k] / p_union[i] - temp * p_diffI[j*4 + k];
            }
        //    std::cout << "diff is " << p_diffI[j*4] << " " << p_diffI[j*4 + 1] << " " << p_diffI[j*4 + 2] << " " << p_diffI[j*4 + 3] << std::endl;
        }

      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_I_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(IouLossLayer);
#endif

INSTANTIATE_CLASS(IouLossLayer);
REGISTER_LAYER_CLASS(IouLoss);

}  // namespace caffe
