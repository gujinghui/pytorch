#include <caffe2/ideep/ideep_utils.h>
#include "caffe2/operators/expand_squeeze_dims_op.h"

namespace caffe2 {

class IDEEPExpandDimsOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPExpandDimsOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        dims_(OperatorBase::GetRepeatedArgument<int>("dims")) {
    auto originalSize = dims_.size();
    CAFFE_ENFORCE(originalSize > 0, "Parameter `dims` must be provided.");
    std::sort(dims_.begin(), dims_.end());
    dims_.erase(std::unique(dims_.begin(), dims_.end()), dims_.end());
    if (dims_.size() < originalSize) {
      LOG(WARNING) << "Parameter `dims` has repeated dimensions.";
    }
    CAFFE_ENFORCE(dims_.front() >= 0, "Dimension ids must be non-negative.");
  }

  bool RunOnDevice() override {
    if (OperatorBase::InputBlob(INPUT).template IsType<itensor>()) {
      return RunWithIDEEPTensor();
    }

    return RunWithCPUTensor();
  }

  bool RunWithIDEEPTensor() {
    const auto& X = Input(INPUT);
    auto* Y = Output(OUTPUT);
    if (&X != Y) {
      // Copy if not inplace
      ideep::direct_copy::compute(X, *Y);
    }
    if (dims_.empty()) {
      return true;
    }

    auto newDims = X.get_dims();
    CAFFE_ENFORCE_GE(
        newDims.size() + dims_.size(),
        dims_.back() + 1,
        "Input needs at least ",
        (1 + dims_.back() - dims_.size()),
        " dimensions given `dims`.");

    for (const auto dim : dims_) {
      newDims.insert(newDims.begin() + dim, 1);
    }

    Y->reshape(newDims);
    return true;
  }

  bool RunWithCPUTensor() {
    const auto& X = OperatorBase::Input<Tensor>(INPUT, CPU);
    auto* Y = OperatorBase::Output<Tensor>(OUTPUT, CPU);
    Y->CopyFrom(X, &context_);
    if (dims_.empty()) {
      return true;
    }

    auto newDims = X.sizes().vec();
    CAFFE_ENFORCE_GE(
        newDims.size() + dims_.size(),
        dims_.back() + 1,
        "Input needs at least ",
        (1 + dims_.back() - dims_.size()),
        " dimensions given `dims`.");

    for (const auto dim : dims_) {
      newDims.insert(newDims.begin() + dim, 1);
    }

    Y->Reshape(newDims);
    return true;
  }

 private:
  vector<int> dims_;

  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT);
};


class IDEEPSqueezeOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPSqueezeOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        dims_(OperatorBase::GetRepeatedArgument<int>("dims")) {
    auto originalSize = dims_.size();
    CAFFE_ENFORCE(originalSize > 0, "Parameter `dims` must be provided.");

    std::sort(dims_.begin(), dims_.end());
    dims_.erase(std::unique(dims_.begin(), dims_.end()), dims_.end());
    if (dims_.size() < originalSize) {
      LOG(WARNING) << "Parameter `dims` has repeated dimensions.";
    }
    CAFFE_ENFORCE(dims_.front() >= 0, "Dimension ids must be non-negative.");
  }

  virtual ~IDEEPSqueezeOp() {}

  bool RunOnDevice() override {
    if (OperatorBase::InputBlob(INPUT).template IsType<itensor>()) {
      return RunWithIDEEPTensor();
    }

    return RunWithCPUTensor();
  }

  bool RunWithIDEEPTensor() {
    const auto& X = Input(INPUT);
    auto* Y = Output(OUTPUT);

    CAFFE_ENFORCE_GT(
        X.ndims(),
        dims_.back(),
        "Input needs at least ",
        (dims_.back() + 1),
        " dimensions.");
    const auto& ideep_dims = X.get_dims();
    vector<int64_t> dims(ideep_dims.begin(), ideep_dims.end());
    const auto& new_dims = SqueezeOp<IDEEPContext>::ComputeDims(dims, dims_);
    itensor::dims new_dims_ideep(new_dims.begin(), new_dims.end());
    if (&X != Y) {
      // Copy if not inplace
      ideep::direct_copy::compute(X, *Y);
    }

    Y->reshape(new_dims_ideep);
    return true;
  }

  bool RunWithCPUTensor() {
    const auto& X = OperatorBase::Input<Tensor>(INPUT, CPU);
    auto* Y = OperatorBase::Output<Tensor>(OUTPUT, CPU);
    Y->CopyFrom(X, &context_);

    CAFFE_ENFORCE_GT(
        X.dim(),
        dims_.back(),
        "Input needs at least ",
        (dims_.back() + 1),
        " dimensions.");

    std::vector<int> newDims =
      SqueezeOp<IDEEPContext>::ComputeDims(X.sizes(), dims_);
    Y->Reshape(newDims);
    return true;
  }

 private:
  vector<int> dims_;

  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT);
};


REGISTER_IDEEP_OPERATOR(ExpandDims, IDEEPExpandDimsOp);
REGISTER_IDEEP_OPERATOR(Squeeze, IDEEPSqueezeOp);

} // namespace caffe2
