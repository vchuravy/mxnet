/*!
 * Copyright (c) 2015 by Contributors
 * \file batch_norm.cc
 * \brief
 * \author Bing Xu
*/

#include "./batch_norm-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(BatchNormParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new BatchNormOp<cpu, DType>(param);
  });
  return op;
}

Operator *BatchNormProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                              std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(BatchNormParam);

MXNET_REGISTER_OP_PROPERTY(BatchNorm, BatchNormProp)
.describe("Apply batch normalization to input.")
.add_argument("data", "Symbol", "Input data to batch normalization")
.add_arguments(BatchNormParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

