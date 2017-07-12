/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "Layer.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

/**
 * A layer for each row of a matrix, multiplying with a element of a vector,
 * which is used in NEURAL TURING MACHINE.
 * \f[
 *   y.row[i] = w[i] * x.row[i]
 * \f]
 * where \f$x\f$ is (batchSize x dataDim) input, \f$w\f$ is
 * (batchSize x 1) weight vector, and \f$y\f$ is (batchSize x dataDim) output.
 *
 * The config file api is scaling_layer.
 */

class ScaleDotAttLayer : public Layer {
public:
  explicit ScaleDotAttLayer(const LayerConfig& config) : Layer(config), _scale(1.0), _mask_strategy(0) {}

  ~ScaleDotAttLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
private:
  std::vector<MatrixPtr> _qk_dots;
  real _scale;
  int _mask_strategy;
};

REGISTER_LAYER(scale_dot_att, ScaleDotAttLayer);

bool ScaleDotAttLayer::init(const LayerMap& layerMap,
                        const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  // required: q, k, v. Optional: mask
  CHECK_GE(inputLayers_.size(), 3U);
  _mask_strategy = config_.mask_strategy();

  return true;
}

void ScaleDotAttLayer::forward(PassType passType) {
  /*
  suppose source sequence length is 4, target length is 6, both head_size = 6;
  for encoder:
    - Q = 4 * 6
    - K = 4 * 6
    - Q * K = 4 * 4
    - V = 4 * 6
    - scale out is 4 * 6
  for decoder:
    - Q = 6 * 6
    - K = 4 * 6
    - V = 4 * 6 
    - Q * K = 6 * 4
    - scale out is 6 * 6
  */
  Layer::forward(passType);
  std::ostringstream os;
  auto debug_matrix = [&os](MatrixPtr m, std::string matrix_name) {
    LOG(INFO) << "Joe, forward pass\n" << matrix_name << ": " \
     << m->getHeight() << ":" << m->getWidth();
    // LOG(INFO) << "Joe, forward pass\n" << matrix_name << "(" << (m->getData()) << "): " \
    //  << m->getHeight() << ":" << m->getWidth();
    m->print(os);
    LOG(INFO) << os.str() << "\n++++++++++++++++++++++++++++++++++";
  };

  auto Q = getInput(0);
  auto K = getInput(1);
  auto V = getInput(2);
  MatrixPtr Q_val = getInputValue(0);
  MatrixPtr K_val = getInputValue(1);
  MatrixPtr V_val = getInputValue(2);
  if (config_.scale()) {
    _scale = 1.0f / sqrt(K_val->getWidth());
  }
  reserveOutput(Q_val->getHeight(), V_val->getWidth());

  auto q_start_positions = Q.sequenceStartPositions->getData(false);
  auto k_start_positions = K.sequenceStartPositions->getData(false);
  auto v_start_positions = V.sequenceStartPositions->getData(false);
  size_t q_num_sequences = Q.getNumSequences();
  size_t k_num_sequences = K.getNumSequences(); 
  size_t v_num_sequences = V.getNumSequences();
  auto output_val = getOutputValue();
  CHECK(q_num_sequences > 0);
  CHECK_EQ(v_num_sequences, k_num_sequences);
  CHECK_EQ(q_num_sequences, k_num_sequences);

  _qk_dots.clear();

  // 1. q matmul k
  for (size_t seq_id = 0; seq_id < q_num_sequences; ++seq_id) {
    size_t q_end_pos = q_start_positions[seq_id + 1];
    size_t q_seq_len = q_end_pos - q_start_positions[seq_id];
    size_t k_end_pos = k_start_positions[seq_id + 1];
    size_t k_seq_len = k_end_pos - k_start_positions[seq_id];

    auto current_q = Q_val->subRowMatrix(q_start_positions[seq_id], q_end_pos);
    auto current_k = K_val->subRowMatrix(k_start_positions[seq_id], k_end_pos);

    auto qk_dot = Matrix::create(q_seq_len, k_seq_len, false, useGpu_);
    if (current_q->getWidth() != current_k->getWidth()) {
      debug_matrix(current_q, "current_q-01");
      debug_matrix(current_k, "current_k-02");
      debug_matrix(qk_dot, "qk_dot-02");
    }
    qk_dot->mul(*current_q, *current_k->getTranspose(), _scale, 0.0);

    for (size_t r = 0; r < q_seq_len; ++r) {
      auto current_row = qk_dot->subRowMatrix(r, r + 1);
      // auto row_mask = Matrix::create(1, q_seq_len, false, useGpu_);
      if (_mask_strategy > 0 && r < q_seq_len - 1) {
        CHECK_EQ(q_seq_len, k_seq_len) << 
        "Currently, only self-attention supports mask, i.e. sequence length of Q and K must equal";
        // perform mask by adding big negative bias
        // row_mask->zeroMem();
        auto masked_pos = Matrix::create(current_row->getData() + r + 1, 1, q_seq_len - r - 1, false, useGpu_);
        masked_pos->addScalar(*masked_pos, -1e9);
        //current_row -> add(*row_mask, 1.0f);
      }
      // do softmax
      current_row->softmax(*current_row);
    }
    // debug_matrix(qk_dot, "first_time_qk_dot");
    _qk_dots.push_back(qk_dot);
  }  

  CHECK_EQ(_qk_dots.size(), v_num_sequences);

  // 5. qk_dot matmul v
  size_t current_pos = 0;
  for (size_t seq_id = 0; seq_id < v_num_sequences; ++seq_id) {
    auto qk_dot = _qk_dots[seq_id];
    auto current_v = V_val->subRowMatrix(v_start_positions[seq_id],
     v_start_positions[seq_id + 1]);
    auto att_v = Matrix::create(output_val->getData() + current_pos,
     qk_dot->getHeight(), V_val->getWidth(), false, useGpu_);

    if (qk_dot->getWidth() != current_v->getHeight()) {
      debug_matrix(qk_dot, "qk_dot-03");
      debug_matrix(current_v, "current_v-04");
      debug_matrix(att_v, "att_v-04");
    }
    att_v->mul(*qk_dot, *current_v, 1, 0.0);
    // debug_matrix(att_v, "att_v-04");
    current_pos += qk_dot->getHeight() * V_val->getWidth();
  }
}

void ScaleDotAttLayer::backward(const UpdateCallback& callback) {
  /*
    suppose source sequence length is 4, target length is 6, both head_size = 6;
    for encoder:
      - Q = 4 * 6
      - K = 4 * 6
      - Q * K = 4 * 4
      - V = 4 * 6
      - scale out is 4 * 6

      backward pass, output_grad size equal to scale_out
        - gradient of V:
          - qk_dot(4*4)->T() * output_grad(4*6)
        - gradient of Q:
          - qk_dot(4*4) * K(4*6)
        - gradient of K:
          - qk_dot(4*4)->T() * Q(4*6)
    for decoder:
      - Q = 6 * 6
      - K = 4 * 6
      - V = 4 * 6 
      - Q * K = 6 * 4
      - scale out is 6 * 6

      backward pass,
        - gradient of qk_dot:
          - output_grad(6*6) * V(4*6)->T() -> 6*4
        - gradient of V:
          - qk_dot(6*4)->T() * output_grad(6*6) -> 4*6
        - gradient of Q:
          - qk_dot(6*4) * K(4*6) -> 6*6
        - gradient of K:
          - qk_dot(6*4)->T() * Q(6*6) -> 4*6
  */
  auto Q = getInput(0);
  auto K = getInput(1);
  auto V = getInput(2);

  auto Q_grad = getInputGrad(0);
  auto K_grad = getInputGrad(1);
  auto V_grad = getInputGrad(2);
  auto out_grad = getOutputGrad();

  MatrixPtr Q_val = getInputValue(0);
  MatrixPtr K_val = getInputValue(1);
  MatrixPtr V_val = getInputValue(2);

  auto v_start_positions = V.sequenceStartPositions->getData(false);
  auto q_start_positions = Q.sequenceStartPositions->getData(false);
  auto k_start_positions = K.sequenceStartPositions->getData(false);
  size_t v_num_sequences = V.getNumSequences();

  std::ostringstream os;
  auto debug_matrix = [&os](MatrixPtr m, std::string matrix_name) {
    LOG(INFO) << "Joe, check_test backward pass\n" << matrix_name << "(" << (m->getData()) << "): " \
     << m->getHeight() << ":" << m->getWidth() << ":" << m->isTransposed();
    m->print(os);
    LOG(INFO) << os.str() << "\n++++++++++++++++++++++++++++++++++";
  };
  int debug_count = 0;
  auto debug_point = [&debug_count](int pos) {
    LOG(INFO) << "Joe, backward pass debug point [" << debug_count++ << "]=" << pos;
  };

  bool debug_flag = false;
  if (debug_flag) {
    //Joe: for gradient checking
    out_grad->resetOne();
    debug_point(0);
    debug_matrix(out_grad, "output_gradient");
    LOG(INFO) << "Joe: backward reset gradient to check";
  }
  // 1. gradient of V
  std::vector<MatrixPtr> _qk_dots_grad;
  for (size_t seq_id = 0; seq_id < v_num_sequences; ++seq_id) {
    // output gradient
    MatrixPtr current_out_grad = out_grad->subRowMatrix(q_start_positions[seq_id],
     q_start_positions[seq_id + 1]);
    auto qk_dot = _qk_dots[seq_id];

    // calculate gradient of v
    auto current_v_grad = V_grad->subRowMatrix(v_start_positions[seq_id],
     v_start_positions[seq_id + 1]);
    auto current_v = V_val->subRowMatrix(v_start_positions[seq_id],
     v_start_positions[seq_id + 1]);

    current_v_grad->mul(*qk_dot->getTranspose(), *current_out_grad);
    // debug_matrix(current_v_grad, "v_grad");

    auto qk_dot_grad = Matrix::create(qk_dot->getHeight(), qk_dot->getWidth(), false, useGpu_);
    _qk_dots_grad.push_back(qk_dot_grad);
    // debug_matrix(qk_dot_grad, "qk_dot_grad-4");
    qk_dot_grad->mul(*current_out_grad, *current_v->getTranspose(), 1, 0.0);
    // debug_matrix(qk_dot_grad, "qk_dot_grad");
  }

  // 2. gradient of k, q
  for (size_t seq_id = 0; seq_id < v_num_sequences; ++seq_id) {
    auto qk_dot = _qk_dots[seq_id];
    auto qk_dot_g = _qk_dots_grad[seq_id];
    size_t q_num = qk_dot->getHeight();

    // softmax derivative, implementated according to softmax activation
    for (size_t r = 0; r < q_num; ++r) {
      auto current_row = qk_dot->subRowMatrix(r, r + 1);
      auto current_row_grad = qk_dot_g->subRowMatrix(r, r + 1);
      if (useGpu_) {
        current_row_grad->softmaxBackward(*current_row);
      } else {
        MatrixPtr sftMaxDot = Matrix::create(1, current_row->getWidth(),
         false, useGpu_);
        MatrixPtr sftMaxSum = Matrix::create(1, 1, false, useGpu_);
        // debug_matrix(sftMaxDot, "sftMaxDot-dotMul-5");
        sftMaxDot->dotMul(*current_row_grad, *current_row);
        sftMaxSum->colMerge(*sftMaxDot);

        current_row_grad->softmaxDerivative(*current_row, *sftMaxSum);
        //TODO: check. Seems no difference for the order of appling (scaling) and (mask)
        current_row_grad->mulScalar(_scale);
      }
    }

    // debug_matrix(qk_dot_g, "qk_dot_pre_grad");
 
    // finally, gradient of q and k
    auto current_q_grad = Q_grad->subRowMatrix(q_start_positions[seq_id],
     q_start_positions[seq_id + 1]);
    auto current_q = Q_val->subRowMatrix(q_start_positions[seq_id],
     q_start_positions[seq_id + 1]);

    auto current_k_grad = K_grad->subRowMatrix(k_start_positions[seq_id],
     k_start_positions[seq_id + 1]);
    auto current_k = K_val->subRowMatrix(k_start_positions[seq_id],
     k_start_positions[seq_id + 1]);

    current_q_grad->mul(*qk_dot_g, *current_k, 1, 0.0);
    // debug_matrix(current_q_grad, "q_grad");

    current_k_grad->mul(*qk_dot_g->getTranspose(), *current_q, 1, 0.0);
    // debug_matrix(current_k_grad, "k_grad");
  }

}

}  // namespace paddle
