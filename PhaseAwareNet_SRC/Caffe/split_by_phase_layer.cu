#include <cfloat>
#include <vector>

#include "caffe/layers/split_by_phase_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SplitByPhaseForward(const int nthreads,
    const Dtype* const bottom_data, const int num_filters, Dtype* const top_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int w = index % 64;
        const int h = (index / 64) % 64;
        const int p = (index / 64 / 64) % 64;
        const int c = (index / 64 / 64 / 64) % num_filters;
        const int n = index / 64 / 64 / 64 / num_filters;
        const int  source_index = ((w*8)+(h*8*512)+(p%8)+(p/8)*512)+((n*num_filters+c)*512*512);
        top_data[index] = bottom_data[source_index];;
    }
}

template <typename Dtype>
__global__ void SplitByPhaseForwardSlow(const int nthreads,
    const Dtype* const bottom_data, const int num_filters, Dtype* const top_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int h, p, c, n, source_index;
        for (int w = 0; w < 64; ++w) {
            h = index % 64;
            p = (index / 64) % 64;
            c = (index / 64 / 64) % num_filters;
            n = index / 64 / 64 / num_filters;
            source_index = ((w*8)+(h*8*512)+(p%8)+(p/8)*512)+((n*num_filters+c)*512*512);
            top_data[index*64+w] = bottom_data[source_index];
        }
    }
}

template <typename Dtype>
void SplitByPhaseLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int count = top[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SplitByPhaseForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, num_filters_, top_data);
}

template <typename Dtype>
__global__ void SplitByPhaseBackwardSlow(const int nthreads,
    Dtype* const bottom_diff, const int num_filters, const Dtype* const top_diff) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int w = index % 64;
        const int h = (index / 64) % 64;
        const int p = (index / 64 / 64) % 64;
        const int c = (index / 64 / 64 / 64) % num_filters;
        const int n = index / 64 / 64 / 64 / num_filters;
        const int  source_index = ((w*8)+(h*8*512)+(p%8)+(p/8)*512)+((n*num_filters+c)*512*512);
        bottom_diff[source_index] = top_diff[index];
    }
}

template <typename Dtype>
__global__ void SplitByPhaseBackward(const int nthreads,
    Dtype* const bottom_diff, const int num_filters, const Dtype* const top_diff) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int w = index % 512;
        const int h = (index / 512) % 512;
        const int c = (index / 512 / 512) % num_filters;
        const int n = index / 512 / 512 / num_filters;
        const int  target_index = ((w/8)+64*(h/8))+(64*64*(((w%8)+8*(h%8))))+(512*512*(n*num_filters+c));
        bottom_diff[index] = top_diff[target_index];
    }
}

template <typename Dtype>
void SplitByPhaseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]) {
        const Dtype* top_diff = top[0]->gpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        const int count = bottom[0]->count();
        // NOLINT_NEXT_LINE(whitespace/operators)
        SplitByPhaseBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_diff, num_filters_, top_diff);
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(SplitByPhaseLayer);

}  // namespace caffe
