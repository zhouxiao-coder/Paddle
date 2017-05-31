#pragma once

#include "Layer.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/ThreadLocal.h"
#include "MultinomialSampler.h"

namespace paddle {

/**
 * @brief The SampledFullyConnectedLayer class
 * SampledFullyConnectedLayer samples negative samples against positive samples.
 * It can be useful to implement various sampling methods, such as importance sampling, 
 * noise contrastive estimation, and negative sampling.
 *
 * The config file api is sampled_fc.
 */
class SampledFullyConnectedLayer : public Layer {
protected:
  WeightList weights_;
  std::unique_ptr<Weight> biases_;

private:
  MatrixPtr mmat_;
  /// cpuSelCols_ is a CpuSparseMatrix, used to save selected columns.
  MatrixPtr cpuSelCols_;
  /// CpuSparseMatrix or GpuSparseMatrix. In CPU mode, selCols_ points
  /// to cpuSelCols_.
  MatrixPtr selCols_;

  MatrixPtr cpuSelColsProbs_;
  MatrixPtr selColsProbs_;

  size_t inputNum_;
  size_t _fullOutputSize;

  /// interOutput_ shared same memory with output_.value.
  MatrixPtr interOutput_;

  /// interOutGrad_ sparse matrix
  MatrixPtr interOutGrad_;

  LayerPtr labelLayer_;
  std::unique_ptr<MultinomialSampler> sampler_;

public:
  explicit SampledFullyConnectedLayer(const LayerConfig& config)
      : Layer(config), selCols_(nullptr) {}

  ~SampledFullyConnectedLayer() {}
  void prefetch();

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  Weight& getWeight(int idx) { return *weights_[idx]; }

  /**
   * @brief Resize the output matrix size.
   * And reset value to zero
   */
  void reserveOutput(size_t height, size_t width, size_t nnz);

  void forward(PassType passType);
  void backward(const UpdateCallback& callback = nullptr);

private:
  void prepareSamples();
};
}  // namespace paddle