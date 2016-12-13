/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#include "VariationalNormLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "paddle/math/SparseMatrix.h"
#include "paddle/math/Matrix.h"
#include "paddle/math/BaseMatrix.h"
#include <vector>
#include <algorithm>

namespace paddle {

REGISTER_LAYER(vae_norm, VariationalNormLayer);

bool VariationalNormLayer::init(const LayerMap& layerMap,
                               const ParameterMap& parameterMap) {
    /* Initialize the basic parent class */
    Layer::init(layerMap, parameterMap);
    /* initialize the weightList */
    CHECK(inputLayers_.size() == parameters_.size());

    mean_weights_.reserve(inputLayers_.size());
    var_weights_.reserve(inputLayers_.size());
    for (size_t i = 0; i < inputLayers_.size(); i++) {
        size_t height = inputLayers_[i]->getSize();
        size_t width = getSize();
        if (parameters_[i]->isSparse()) {
            CHECK_LE(parameters_[i]->getSize(), 2 * width * height);
        } else {
            CHECK_EQ(parameters_[i]->getSize(), 2 * width * height);
        }
        mean_weights_.emplace_back(new Weight(height, width, parameters_[i], 0));
        var_weights_.emplace_back(new Weight(height, width, parameters_[i], height * width));
    }

    /* initialize mean and variance biases */
    if (biasParameter_.get() != NULL) {
        mean_biases_ = std::unique_ptr<Weight>(new Weight(1, getSize(), biasParameter_, 0));
        var_biases_ = std::unique_ptr<Weight>(new Weight(1, getSize(), biasParameter_, getSize()));
    }
    // means and variances result
    mean_ = Matrix::create(nullptr, /*height= */ 1, getSize(), /*trans= */ false, useGpu_);
    var_ = Matrix::create(nullptr, /*height= */ 1, getSize(), /*trans= */ false, useGpu_);

    return 0;
}


void VariationalNormLayer::forward(PassType passType) {
    Layer::forward(passType);

    /* malloc memory for the output_ if necessary */
    int batchSize = getInput(0).getBatchSize();
    int size = getSize();
    {
        REGISTER_TIMER_INFO("VAENormResetTimer", getName().c_str());
        reserveOutput(batchSize, size);
    }

    Matrix::resizeOrCreate(mean_, /* height= */ batchSize, getSize(),
            /* trans= */ false, useGpu_);
    Matrix::resizeOrCreate(var_, /* height= */ batchSize, getSize(),
            /* trans= */ false, useGpu_);
    MatrixPtr outV = getOutputValue();
    // 1. compute mean and variance
    for (size_t i = 0; i != inputLayers_.size(); ++i) {
        auto input = getInput(i);
        CHECK(input.value) << "The input of 'vae_norm' layer must be matrix";
        REGISTER_TIMER_INFO("VAENormMulTimer", getName().c_str());
        i == 0 ? mean_->mul(input.value, mean_weights_[i]->getW(), 1, 0)
               : mean_->mul(input.value, mean_weights_[i]->getW(), 1, 1);
        i == 0 ? var_->mul(input.value, var_weights_[i]->getW(), 1, 0)
               : var_->mul(input.value, var_weights_[i]->getW(), 1, 1);
    }

//    real debug_var = var_weights_[0]->getW()->getData()[0];
//    LOG(INFO) << "Joe: foward pass, var_weights_[0][0]=" << debug_var;

    if (mean_biases_.get() != NULL) {
        REGISTER_TIMER_INFO("VAENormBiasTimer", getName().c_str());
        mean_->addBias(*(mean_biases_->getW()), 1);
        var_->addBias(*(var_biases_->getW()), 1);
    }

//    LOG(INFO) << "Joe: 2. reparameterization";
    // 2. compute output `z` using the "reparameterization trick"
    // outV z_i = z_mu_i + kesi_i * sqrt(exp(z_ls2_i)).
    VectorPtr random_value = Vector::create(getSize() * batchSize, useGpu_);
    random_value->randnorm(0, 1);
    MatrixPtr kesi = Matrix::create(random_value->getData(), batchSize, getSize(),
        /*trans = */ false, useGpu_);
    Matrix::resizeOrCreate(last_kesi_, /* height= */ batchSize, getSize(),
      /* trans= */ false, useGpu_);
    last_kesi_->copyFrom(*kesi);

//    LOG(INFO) << "Joe: 3. exp->sqrt->mul->copyFrom";
    var_->exp();
    var_->sqrt();
    mean_->addDotMul(*kesi, *var_, 1.0, 1.0);
    outV->copyFrom(*mean_);

//    /* activation */ {
//        REGISTER_TIMER_INFO("FwAtvTimer", getName().c_str());
//        forwardActivation();
//    }
}

/*
 *  z_i = z_mean_i + kesi_i * sqrt(exp(z_var_i)).
 *  z_mean_ = w_mean * x + b_mean
 *  z_var_ = w_var * x + b_var
 *
 *  Gradient:
 *
 *  - z_mean_/w_mean = x
 *  - z_mean_/b_mean = 1
 *  - z_var_/w_var = 0.5 * kesi * sqrt(e^var) * x
 *  - z_var_/b_var = 0.5 * kesi * sqrt(e^var)
 *
 *  - z/x = w_mean + 0.5 * kesi * sqrt(e^var) * w_var
 * */
void VariationalNormLayer::backward(const UpdateCallback& callback) {
//    /* Do derivation */ {
//        REGISTER_TIMER_INFO("BpAvtTimer", getName().c_str());
//        backwardActivation();
//    }

    int batchSize = getInput(0).getBatchSize();
//    LOG(INFO) << "Joe backward: 1. varGrad";
    MatrixPtr varGrad = Matrix::create(batchSize, getSize(), false, useGpu_);
    varGrad->addDotMul(*last_kesi_, *var_, 0.5, 0.0);
//    varGrad->mul(last_kesi_, var_, 0.5, 0.0);
    varGrad->dotMul(*getOutputGrad());
//    varGrad->rightMul(*getOutputGrad());

//    LOG(INFO) << "Joe backward: 2. w_bias";
    if (mean_biases_ && mean_biases_->getWGrad()) {
        REGISTER_TIMER_INFO("BpBiasTimer", getName().c_str());
        mean_biases_->getWGrad()->collectBias(*getOutputGrad(), 1);
        var_biases_->getWGrad()->collectBias(*varGrad, 1);
        /* Increasing the number of gradient */
//        mean_biases_->getParameterPtr()->incUpdate(callback);
        var_biases_->getParameterPtr()->incUpdate(callback);
    }

    bool syncFlag = hl_get_sync_flag();
    // TODO: check parameter gradient
//    LOG(INFO) << "Joe backward: 3. w_mean and input ";
    for (size_t i = 0; i != inputLayers_.size(); ++i) {
        /* Calculate the W-gradient for the current layer */
        if (mean_weights_[i]->getWGrad()) {
//            LOG(INFO) << "Joe backward: 3.1 w_mean get";
            MatrixPtr input_T = getInputValue(i)->getTranspose();
            MatrixPtr oGrad = getOutputGrad();
//            LOG(INFO) << "Joe backward: 3.1 w_mean cal";
            {
                REGISTER_TIMER_INFO("GradMulTimer", getName().c_str());
                mean_weights_[i]->getWGrad()->mul(input_T, oGrad, 1, 1);
                var_weights_[i]->getWGrad()->mul(input_T, varGrad, 1, 1);
            }
        }
        // If callback does not change value, backprop error asynchronously so that
        // we can do the callback concurrently.
        hl_set_sync_flag(false);

        // TODO: implement gradient z/x
        /* Calculate the input layers error */
//      - z/x = w_mean + 0.5 * kesi * sqrt(e^var) * w_var
      MatrixPtr preGrad = getInputGrad(i);
      if (NULL != preGrad) {
//          LOG(INFO) << "Joe backward: 3.2 input cal get";
          MatrixPtr mean_weights_T = mean_weights_[i]->getW()->getTranspose();
          MatrixPtr var_weights_T = var_weights_[i]->getW()->getTranspose();
//          LOG(INFO) << "Joe backward: 3.2 input cal ";
          preGrad->mul(getOutputGrad(), mean_weights_T, 1, 1);
          preGrad->mul(varGrad, var_weights_T, 1, 1);
          REGISTER_TIMER_INFO("BpMulTimer", getName().c_str());
      }

      hl_set_sync_flag(syncFlag);
      {
          REGISTER_TIMER_INFO("WeightUpdate", getName().c_str());
          // TODO: mean_weights_ and var_weights_ share same parameter,
          // so 2 incUpdate or 1 should be used ?
//            mean_weights_[i]->getParameterPtr()->incUpdate(callback);
          var_weights_[i]->getParameterPtr()->incUpdate(callback);
      }
    }
}

}  // namespace paddle
