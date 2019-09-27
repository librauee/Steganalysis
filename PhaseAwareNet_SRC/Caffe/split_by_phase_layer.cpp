#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/split_by_phase_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SplitByPhaseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    num_images_ = bottom[0]->num();
    num_filters_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();

    CHECK_EQ(height_, 512);
    CHECK_EQ(width_, 512);

}

template <typename Dtype>
void SplitByPhaseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    top[0]->Reshape(num_images_, num_filters_*64, 64, 64);

}

template <typename Dtype>
void SplitByPhaseLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int n, c, p, h, w, source_index;
  for (int index = 0; index < bottom[0]->count(); ++index) {
      w = index % 64;
      h = (index / 64) % 64;
      p = (index / 64 / 64) % 64;
      c = (index / 64 / 64 / 64) % num_filters_;
      n = index / 64 / 64 / 64 / num_filters_;
      source_index = ((w*8)+(h*8*512)+(p%8)+(p/8)*512)+((n*num_filters_+c)*512*512);
      top_data[index] = bottom_data[source_index];
  }
}

template <typename Dtype>
void SplitByPhaseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]) {
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        int n, c, p, h, w, source_index;
        for (int index = 0; index < bottom[0]->count(); ++index) {
            w = index % 64;
            h = (index / 64) % 64;
            p = (index / 64 / 64) % 64;
            c = (index / 64 / 64 / 64) % num_filters_;
            n = index / 64 / 64 / 64 / num_filters_;
            source_index = ((w*8)+(h*8*512)+(p%8)+(p/8)*512)+((n*num_filters_+c)*512*512);
            bottom_diff[source_index] = top_diff[index];
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(SplitByPhaseLayer);
#endif

INSTANTIATE_CLASS(SplitByPhaseLayer);
REGISTER_LAYER_CLASS(SplitByPhase);

}  // namespace caffe
