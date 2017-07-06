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

#include "Operator.h"

namespace paddle {

/**
 * SequenceMulOperator takes two inputs, performs element-wise multiplication:
 * \f[
 *   out.row[i] += scale * (in1.row[i] .* in2.row[i])
 * \f]
 * where \f$.*\f$ means element-wise multiplication,
 * and scale is a config scalar, its default value is one.
 *
 * The config file api is dotmul_operator.
 */
class SequenceMulOperator : public Operator {
public:
  SequenceMulOperator(const OperatorConfig& config, bool useGpu);
  virtual void forward();
  virtual void backward();
};

REGISTER_OPERATOR(seq_mul, SequenceMulOperator);

SequenceMulOperator::SequenceMulOperator(const OperatorConfig& config, bool useGpu)
    : Operator(config, useGpu) {
  CHECK_EQ(config_.input_indices_size(), 2L);
}

void SequenceMulOperator::forward() {

  std::ostringstream os;

  auto v1 = ins_[0]->value;
  auto v2 = ins_[1]->value;
  auto v2t = v2->getTranspose();

  // LOG(INFO) << "Joe: in SeqMul, v1 shape=" << v1->getWidth() << "," \
  //   << v1->getHeight() << "," << v1->getStride() << "," << v1->getElementCnt();
  // LOG(INFO) << "Joe: in SeqMul, transposed v2.t() shape=" << v2t->getWidth() << "," \
  //   << v2t->getHeight() << "," << v2t->getStride() << "," << v2t->getElementCnt();

  size_t numSequences1 = ins_[0]->getNumSequences();
  auto startPositions1 = ins_[0]->sequenceStartPositions->getData(false);

  size_t numSequences2 = ins_[1]->getNumSequences();
  auto startPositions2 = ins_[1]->sequenceStartPositions->getData(false);
  size_t batch_size = ins_[0]->getBatchSize() ;

  // LOG(INFO) << "Joe: numSequences1 = " << numSequences1 << ", numSequences2 = " << numSequences2;
  // ins_[0]->sequenceStartPositions->getVector(false)->print(os, numSequences1);
  // LOG(INFO) << "Joe: seq vectors 1 = " << os.str() << "\t" << "array_version:" << \
  //  startPositions1[0] << "," << startPositions1[1];
  // ins_[1]->sequenceStartPositions->getVector(false)->print(os, numSequences2);
  // LOG(INFO) << "Joe: seq vectors 2 = " << os.str() ;
  // LOG(INFO) << "Joe: batch_size = " << batch_size ;

  /*
  1. pre-calculate size needed: 
  total_size = 0
  for q in qs:
    for k in ks:
      total_size += n * m
      sequenceStartPositions[++i] = n*m + sequenceStartPositions[i]

  subsequenceStartPositions = range(total_size)     
  output->val = Matrix(total_size, 1)
  2. actual calculations
  for q, k in zip(qs, ks):
    subMatrix = q * k
  */
  CHECK(numSequences1 > 0);
  CHECK_EQ(numSequences1, numSequences2);

  size_t total_size = 0;
  ICpuGpuVector::resizeOrCreate(out_->sequenceStartPositions, numSequences1 + 1, false/*useGpu=*/);
  auto output_sequence_info = out_->sequenceStartPositions->getMutableData(false);
  output_sequence_info[0] = 0;
  LOG(INFO) << "address of output_sequence_info " << output_sequence_info ;
  LOG(INFO) << "address of startPositions1 " << startPositions1 ;

  for (size_t seqId = 0; seqId < numSequences1; ++seqId) {
    size_t endPos1 = seqId + 1 >= numSequences1 ? batch_size : startPositions1[seqId + 1];
    size_t seqLen1 = endPos1 > startPositions1[seqId] ? endPos1 - startPositions1[seqId] : 4;
    LOG(INFO) << "Joe: seq vectors 1 = " << os.str() << "\t" << "array_version:" << \
    startPositions1[0] << "," << startPositions1[1] << "," << startPositions1[2];
    LOG(INFO) << "Joe: seqLen1(" << seqLen1 << ")= endPos1(" << endPos1 <<  \
      ") - startPositions1[seqId("<<seqId<<")](" << startPositions1[seqId];


    size_t endPos2 = seqId + 1 >= numSequences2 ? batch_size : startPositions2[seqId + 1];
    size_t seqLen2 = endPos2 - startPositions2[seqId];
    LOG(INFO) << "Joe: seqLen1= " << seqLen1 << ", seqLen2=" << seqLen2 << ", total_size=" << total_size
      << " endPos1, endPos2=" << endPos1 << "," << endPos2;

    total_size += seqLen1 * seqLen2;
    output_sequence_info[seqId + 1] = total_size;
  }  

  LOG(INFO) << "Joe: total_size= " << total_size;
  ICpuGpuVector::resizeOrCreate(out_->subSequenceStartPositions, total_size + 1, false/*useGpu=*/);
  auto output_subsequence_info = out_->subSequenceStartPositions->getMutableData(false);
  output_subsequence_info[0] = 0;
  for (size_t subseqId = 0; subseqId < total_size; ++subseqId) {
    output_subsequence_info[subseqId + 1] = output_subsequence_info[subseqId] + 1;
  }
  Matrix::resizeOrCreate(out_->value, total_size, 1, false, useGpu_);

  // 2nd pass, calculation
  size_t current_start = 0;
  auto outData = out_->value->getData();
  for (size_t seqId = 0; seqId < numSequences1; ++seqId) {
    size_t endPos1 = seqId + 1 >= numSequences1 ? batch_size : startPositions1[seqId + 1];
    size_t seqLen1 = endPos1 - startPositions1[seqId];

    size_t endPos2 = seqId + 1 >= numSequences2 ? batch_size : startPositions2[seqId + 1];
    size_t seqLen2 = endPos2 - startPositions2[seqId];

    auto current_q = v1->subRowMatrix(startPositions1[seqId], endPos1);
    auto current_k = v2->subRowMatrix(startPositions2[seqId], endPos2);

    auto tempMatrix = Matrix::create(outData + current_start, seqLen1, seqLen2, false, useGpu_);
    tempMatrix->mul(*current_q, *current_k->getTranspose(), 1, 0.0);
    current_start += seqLen1 * seqLen2;
  }  



  // out_->value->addDotMul(
  //    *ins_[0]->value, *ins_[1]->value, 1, config_.dotmul_scale());

  // out_->value->resize(ins_[0]->value->getHeight(), ins_[1]->value->getHeight());
  // out_->value->mul(
  //     *ins_[0]->value, *ins_[1]->value->getTranspose(), 1, config_.dotmul_scale());
}

void SequenceMulOperator::backward() {
  const MatrixPtr& inV0 = ins_[0]->value;
  const MatrixPtr& inV1 = ins_[1]->value;
  const MatrixPtr& inG0 = ins_[0]->grad;
  const MatrixPtr& inG1 = ins_[1]->grad;

  if (inG0) {
    inG0->addDotMul(*out_->grad, *inV1, 1, config_.dotmul_scale());
  }
  if (inG1) {
    inG1->addDotMul(*out_->grad, *inV0, 1, config_.dotmul_scale());
  }
}

}  // namespace paddle
