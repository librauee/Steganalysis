#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/split_by_phase_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class SplitByPhaseLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SplitByPhaseLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 1, 512, 512)),
        blob_bottom_nobatch_(new Blob<Dtype>(1, 2, 3, 4)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
//    filler_param.s
    filler_param.set_min(-9);
    filler_param.set_max(9);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~SplitByPhaseLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_nobatch_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_nobatch_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SplitByPhaseLayerTest, TestDtypesAndDevices);

TYPED_TEST(SplitByPhaseLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  shared_ptr<SplitByPhaseLayer<Dtype> > layer(
      new SplitByPhaseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_vec_[0]->num(), 1);
  EXPECT_EQ(this->blob_top_vec_[0]->channels(), 1*64);
  EXPECT_EQ(this->blob_top_vec_[0]->height(), 64);
  EXPECT_EQ(this->blob_top_vec_[0]->width(), 64);
}

TYPED_TEST(SplitByPhaseLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype* bottom_data = this->blob_bottom_->mutable_cpu_data();
  const int num_filters = this->blob_bottom_->channels();
  int n, c, p, h, w, bottom_fill_idx;
  for (int index = 0; index < this->blob_bottom_->count(); ++index) {
      w = index % 64;
      h = (index / 64) % 64;
      p = (index / 64 / 64) % 64;
      c = (index / 64 / 64 / 64) % num_filters;
      n = index / 64 / 64 / 64 / num_filters;
      bottom_fill_idx = ((w*8)+(h*8*512)+(p%8)+(p/8)*512)+((n*num_filters+c)*512*512);
      bottom_data[bottom_fill_idx] = p;
  }
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    shared_ptr<SplitByPhaseLayer<Dtype> > layer(
        new SplitByPhaseLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_vec_[0]->cpu_data();
    const int num_phase_blocks = this->blob_top_vec_[0]->num()*this->blob_top_vec_[0]->channels();
    for (int nc = 0; nc < num_phase_blocks; ++nc) {
        for (int h = 0; h < 64 ; ++h ) {
            for (int w = 0; w < 64 ; ++w ) {
                CHECK_EQ(data[nc*(64*64)+h*64+w], nc%64);
            }
        }
    }
  } else {
         LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(SplitByPhaseLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
//  if (Caffe::mode() == Caffe::CPU ||
//      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
  if (Caffe::mode() == Caffe::GPU) {
    LayerParameter layer_param;
    SplitByPhaseLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe
