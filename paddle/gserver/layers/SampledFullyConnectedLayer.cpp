#include "SampledFullyConnectedLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "paddle/math/SparseMatrix.h"
#include <vector>
#include <set>
#include <algorithm>
#include "paddle/math/MathFunctions.h"

namespace paddle {

REGISTER_LAYER(sampled_fc, SampledFullyConnectedLayer);

bool SampledFullyConnectedLayer::init(const LayerMap& layerMap,
                                        const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);
  _fullOutputSize = config_.full_output_size();

  /* The last input supplies true labels
  */
  inputNum_ = inputLayers_.size() - 1;
  labelLayer_ = inputLayers_[inputNum_];

  for (size_t i = 0; i < inputNum_; i++) {
    size_t height = inputLayers_[i]->getSize();
    size_t width = _fullOutputSize;
    // Notice the weight is transpoed
    weights_.emplace_back(new Weight(width, height, parameters_[i]));
  }

  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, _fullOutputSize, biasParameter_));
  }

  if (config_.neg_sampling_dist_size()) {
    CHECK_EQ(_fullOutputSize, config_.neg_sampling_dist_size());
    // sampler_.reset(new MultinomialSampler(config_.neg_sampling_dist().data(),
                                          // _fullOutputSize));
    sampler_.reset(MultinomialSampler::create(config_.neg_sampling_dist().data(),
                                          _fullOutputSize));
  }

  return true;
}

void SampledFullyConnectedLayer::prefetch() {}

void SampledFullyConnectedLayer::reserveOutput(size_t height, size_t width,
                                                 size_t nnz) {
  SetDevice device(output_.deviceId);
  // output_.value is dense matrix, but width = nnz /height
  CHECK_EQ(nnz % height, 0U);
  CHECK(nnz / height);
  Matrix::resizeOrCreate(output_.value, height, nnz / height,
      /*trans=*/false, /*useGpu=*/useGpu_);
  interOutput_ = Matrix::createSparseMatrix(
      output_.value->getData(), selCols_->getRows(), selCols_->getCols(),
      height, width, nnz, FLOAT_VALUE, SPARSE_CSR,
      /*trans=*/false, /*useGpu=*/useGpu_);

  interOutput_->zeroMem();

  if (passType_ != PASS_TEST && needGradient()) {
    CHECK_EQ(nnz % height, 0U) << "during training, each sample must have a "
                                  "same number of selected columns.";
    CHECK(nnz / height)
        << "during training, "
           "each sample must have at least one column selected.";
    Matrix::resizeOrCreate(output_.grad, height, nnz / height,
                           /*trans=*/false, /*useGpu=*/useGpu_);
    output_.grad->zeroMem();
  }
}

void SampledFullyConnectedLayer::forward(PassType passType) {
  REGISTER_TIMER("selective_fc.forward");
  Layer::forward(passType);

  prepareSamples();
  size_t height = getInput(0).getBatchSize();
  size_t width = _fullOutputSize;

  CHECK(selCols_);
  CHECK(height == selCols_->getHeight());
  CHECK(width == selCols_->getWidth()) << "width=" << width << \
      ", and selCols->getWidth()=" <<  selCols_->getWidth();
  size_t nnz = selCols_->getElementCnt();

  // Layer::ResetOutput(), here we set outV/outG as SparseMatrix manually
  // this outV should be used as input of MaxIdLayer and softmax activation
  reserveOutput(height, width, nnz);

  for (size_t i = 0; i < inputNum_; i++) {
    MatrixPtr input = getInputValue(i);
    MatrixPtr weight = weights_[i]->getW();
    real scaleT = i == 0 ? real(0) : real(1);

    //    always use sparse computation
    REGISTER_TIMER("selective.plain");
    interOutput_->mul(*input, *weight->getTranspose(), 1, scaleT);
  }

  if (biases_) {
    interOutput_->addBias(*(biases_->getW()), 1);
  }

  if (config_.subtract_log_q() > 0) {
    CHECK_EQ(selCols_->getElementCnt(), interOutput_->getElementCnt());
    MatrixPtr probsData = Matrix::create(selCols_->getData(), 1, selCols_->getElementCnt(),
        /*trans=*/false, /*useGpu=*/useGpu_);
    MatrixPtr interOutputData = Matrix::create(interOutput_->getData(), 1, selCols_->getElementCnt(),
        /*trans=*/false, /*useGpu=*/useGpu_);
    interOutputData->add(*probsData);
  }

  forwardActivation();
}

void SampledFullyConnectedLayer::backward(const UpdateCallback& callback) {
  backwardActivation();
  MatrixPtr oGrad = getOutputGrad();
  interOutGrad_ = Matrix::createSparseMatrix(
      oGrad->getData(), interOutput_->getRows(), interOutput_->getCols(),
      interOutput_->getHeight(), interOutput_->getWidth(),
      interOutput_->getElementCnt(), FLOAT_VALUE, SPARSE_CSR,
      /*trans=*/false,
      /*useGpu=*/useGpu_);

  if (biases_ && biases_->getWGrad()) {
    REGISTER_TIMER_INFO("BpBiasTimer", getName().c_str());
    biases_->getWGrad()->collectBias(*interOutGrad_, 1);
    biases_->getParameterPtr()->incUpdate(callback);
  }

  // backward is different from FullyConnectedLayer
  // because the weight is transposed
  for (size_t i = 0; i < inputNum_; i++) {
    AsyncGpuBlock block;
    MatrixPtr preGrad = getInputGrad(i);
    if (preGrad) {
      REGISTER_TIMER_INFO("BpMulTimer", getName().c_str());
      preGrad->mul(*interOutGrad_, *weights_[i]->getW(), 1, 1);
    }

    MatrixPtr wGrad = weights_[i]->getWGrad();
    if (wGrad) {
      REGISTER_TIMER_INFO("GradMulTimer", getName().c_str());
      MatrixPtr input = getInputValue(i);
      wGrad->mul(*interOutGrad_->getTranspose(), *input, 1, 1);
    }

    {
      REGISTER_TIMER_INFO("WeightUpdate", getName().c_str());
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }
  }
}

void SampledFullyConnectedLayer::prepareSamples() {
  size_t batchSize = getInput(*labelLayer_).getBatchSize();
  size_t numberSamples = config_.num_neg_samples();

  //  get label layer information
  IVectorPtr label = getInput(*labelLayer_).ids;
  CHECK(label) << "The label layer must have ids, multi-label is not supported for now";

  auto& randEngine = ThreadLocalRandomEngine::get();
  size_t nnz = batchSize * (1 + config_.num_neg_samples());
  Matrix::resizeOrCreateSparseMatrix(this->cpuSelCols_,
                                     batchSize, _fullOutputSize, nnz,
                                     FLOAT_VALUE, SPARSE_CSR, false, false);
  CHECK(this->cpuSelCols_ != nullptr);
  CpuSparseMatrixPtr selCols = std::dynamic_pointer_cast<CpuSparseMatrix>(cpuSelCols_);
  int* rowOffsets = selCols->getRows();
  int* colIndices = selCols->getCols();
  real* probsData = selCols->getData();

  rowOffsets[0] = 0;
  int idx = 0;

  // sample noise
  std::set<int> noises;
  // hint: "remove accidental hits" can be implemented here, insert true ids into the set 
  // A naive sequential sampling implementation.
  if (config_.share_sample_in_batch()) {
    while(noises.size() < numberSamples) {
      noises.emplace(sampler_->gen(randEngine));
    }
  }

  // update row offsets and true labels for all samples in batch
  for (size_t i = 0; i < batchSize; ++i) {
    // TODO: getElement may be slow for gpu, check vector copy api
    auto true_id = label->getElement(i);
    rowOffsets[i + 1] = rowOffsets[i] + config_.num_neg_samples() + 1;
    colIndices[idx] = true_id;
    if (config_.subtract_log_q() == 1) {
        probsData[idx] = -log(config_.neg_sampling_dist(true_id));
    } else {
        probsData[idx] = -log(config_.num_neg_samples() * config_.neg_sampling_dist(true_id));
    }
    idx++;
    
    if (!config_.share_sample_in_batch()) {
      noises.clear();
      while(noises.size() < numberSamples) {
        noises.emplace(sampler_->gen(randEngine));
      }
    }
    for (auto id: noises) {
      colIndices[idx] = id;
      probsData[idx] = -log(config_.neg_sampling_dist(id));
      idx++;
    }
  }

  if (!useGpu_) {
    this->selCols_ = this->cpuSelCols_;
  } else {
    Matrix::resizeOrCreateSparseMatrix(this->selCols_,
                                       batchSize, _fullOutputSize, nnz,
                                       FLOAT_VALUE, SPARSE_CSR, false, true);
    this->selCols_->copyFrom(*cpuSelCols_, HPPL_STREAM_1);
    hl_stream_synchronize(HPPL_STREAM_1);
  }
}

}  // namespace paddle