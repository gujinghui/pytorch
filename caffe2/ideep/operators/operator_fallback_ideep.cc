#include <caffe2/ideep/operators/operator_fallback_ideep.h>
#include <caffe2/ideep/utils/ideep_operator.h>

#include <caffe2/operators/abs_op.h>
#include <caffe2/operators/atan_op.h>
#include <caffe2/operators/accuracy_op.h>
#include <caffe2/operators/affine_channel_op.h>
#include <caffe2/operators/bbox_transform_op.h>
#include <caffe2/operators/box_with_nms_limit_op.h>
#include <caffe2/operators/channel_shuffle_op.h>
#include <caffe2/operators/cast_op.h>
#include <caffe2/operators/clip_op.h>
#include <caffe2/operators/collect_and_distribute_fpn_rpn_proposals_op.h>
#include <caffe2/operators/conv_transpose_op.h>
#include <caffe2/operators/cross_entropy_op.h>
#include <caffe2/operators/ctc_beam_search_decoder_op.h>
#include <caffe2/operators/ctc_greedy_decoder_op.h>
#include <caffe2/operators/distance_op.h>
#include <caffe2/operators/dropout_op.h>
#include <caffe2/operators/elementwise_add_op.h>
#include <caffe2/operators/elementwise_div_op.h>
#include <caffe2/operators/elementwise_mul_op.h>
#include <caffe2/operators/elementwise_ops.h>
#include <caffe2/operators/elementwise_sub_op.h>
#include <caffe2/operators/expand_op.h>
#include <caffe2/operators/filler_op.h>
#include <caffe2/operators/flatten_op.h>
#include <caffe2/operators/gather_op.h>
#include <caffe2/operators/generate_proposals_op.h>
#include <caffe2/operators/given_tensor_fill_op.h>
#include <caffe2/operators/load_save_op.h>
#include <caffe2/operators/loss_op.h>
#include <caffe2/operators/order_switch_ops.h>
#include <caffe2/operators/pad_op.h>
#include <caffe2/operators/prelu_op.h>
#include <caffe2/operators/roi_align_op.h>
#include <caffe2/operators/roi_align_rotated_op.h>
#include <caffe2/operators/scale_op.h>
#include <caffe2/operators/slice_op.h>
#include <caffe2/operators/softmax_op.h>
#include <caffe2/operators/softmax_with_loss_op.h>
#include <caffe2/operators/sqrt_op.h>
#include <caffe2/operators/stop_gradient.h>
#include <caffe2/operators/tanh_op.h>
#include <caffe2/operators/tensor_protos_db_input.h>
#include <caffe2/operators/transpose_op.h>
#include <caffe2/operators/affine_channel_op.h>
#include <caffe2/operators/stop_gradient.h>
#include <caffe2/operators/slice_op.h>
#include <caffe2/operators/gather_op.h>
#include <caffe2/operators/utility_ops.h>

#include <caffe2/sgd/adam_op.h>
#include <caffe2/sgd/iter_op.h>
#include <caffe2/sgd/learning_rate_op.h>

// can add more non-IDEEP operators if needed
namespace caffe2 {

// Serialization
REGISTER_IDEEP_OPERATOR(
    ChannelShuffle,
    IDEEPFallbackOp<ChannelShuffleOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    LabelCrossEntropy,
    IDEEPFallbackOp<LabelCrossEntropyOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    AveragedLoss,
    IDEEPFallbackOp<AveragedLoss<float, CPUContext>, SkipIndices<0>>);
REGISTER_IDEEP_OPERATOR(
    ConvTranspose,
    IDEEPFallbackOp<ConvTransposeOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(Flatten, IDEEPFallbackOp<FlattenOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(ResizeLike, IDEEPFallbackOp<ResizeLikeOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(Transpose, IDEEPFallbackOp<TransposeOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(Slice, IDEEPFallbackOp<SliceOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(Gather, IDEEPFallbackOp<GatherOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    TensorProtosDBInput,
    IDEEPFallbackOp<TensorProtosDBInput<CPUContext>>);

// Boolean operators
REGISTER_IDEEP_COMPARE_OPERATOR(EQ);
REGISTER_IDEEP_COMPARE_OPERATOR(GT);
REGISTER_IDEEP_COMPARE_OPERATOR(GE);
REGISTER_IDEEP_COMPARE_OPERATOR(LT);
REGISTER_IDEEP_COMPARE_OPERATOR(LE);
REGISTER_IDEEP_COMPARE_OPERATOR(NE);

// Unary element-wise
REGISTER_IDEEP_OPERATOR(
    Abs,
    IDEEPFallbackOp<UnaryElementwiseOp<
      TensorTypes<float>, CPUContext, AbsFunctor<CPUContext>>>);
REGISTER_IDEEP_OPERATOR(
    Atan,
    IDEEPFallbackOp<UnaryElementwiseOp<
      TensorTypes<float>, CPUContext, AtanFunctor<CPUContext>>>);
REGISTER_IDEEP_OPERATOR(
    Sqrt,
    IDEEPFallbackOp<UnaryElementwiseOp<
      TensorTypes<float>, CPUContext, SqrtFunctor<CPUContext>>>);
REGISTER_IDEEP_OPERATOR(
    Tanh,
    IDEEPFallbackOp<UnaryElementwiseOp<
      TensorTypes<float>, CPUContext, TanhFunctor<CPUContext>>>);

// Binary element-wise
REGISTER_IDEEP_OPERATOR(
    Div,
    IDEEPFallbackOp<BinaryElementwiseOp<
      NumericTypes, CPUContext, DivFunctor<CPUContext>>>);
REGISTER_IDEEP_OPERATOR(
    Mul,
    IDEEPFallbackOp<BinaryElementwiseOp<
      NumericTypes, CPUContext, MulFunctor<CPUContext>>>);
REGISTER_IDEEP_OPERATOR(
    Sub,
    IDEEPFallbackOp<BinaryElementwiseOp<
      NumericTypes, CPUContext, SubFunctor<CPUContext>>>);
REGISTER_IDEEP_OPERATOR(
    TanhGradient,
    IDEEPFallbackOp<BinaryElementwiseOp<
      TensorTypes<float>, CPUContext, TanhGradientFunctor<CPUContext>>>);

// Tensor operators
REGISTER_IDEEP_OPERATOR(
    Cast,
    IDEEPFallbackOp<CastOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    Clip,
    IDEEPFallbackOp<ClipOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    Expand,
    IDEEPFallbackOp<ExpandOp<
      TensorTypes<std::int32_t, std::int64_t, float, double>, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    Flatten,
    IDEEPFallbackOp<FlattenOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    Gather,
    IDEEPFallbackOp<GatherOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    NHWC2NCHW,
    IDEEPFallbackOp<NHWC2NCHWOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    NCHW2NHWC,
    IDEEPFallbackOp<NCHW2NHWCOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    ResizeLike,
    IDEEPFallbackOp<ResizeLikeOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    Scale,
    IDEEPFallbackOp<ScaleOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    ScatterAssign,
    IDEEPFallbackOp<ScatterAssignOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    Slice,
    IDEEPFallbackOp<SliceOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    Transpose,
    IDEEPFallbackOp<TransposeOp<CPUContext>>);

// Filter operators
REGISTER_IDEEP_OPERATOR(
    ConstantFill,
    IDEEPFallbackOp<ConstantFillOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    GaussianFill,
    IDEEPFallbackOp<GaussianFillOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    GivenTensorFill,
    IDEEPFallbackOp<GivenTensorFillOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    GivenTensorBoolFill,
    IDEEPFallbackOp<GivenTensorFillOp<bool, CPUContext>, SkipIndices<0>>);
REGISTER_IDEEP_OPERATOR(
    GivenTensorDoubleFill,
    IDEEPFallbackOp<GivenTensorFillOp<double, CPUContext>, SkipIndices<0>>);
REGISTER_IDEEP_OPERATOR(
    GivenTensorIntFill,
    IDEEPFallbackOp<GivenTensorFillOp<int, CPUContext>, SkipIndices<0>>);
REGISTER_IDEEP_OPERATOR(
    GivenTensorInt64Fill,
    IDEEPFallbackOp<GivenTensorFillOp<int64_t, CPUContext>, SkipIndices<0>>);
REGISTER_IDEEP_OPERATOR(
    GivenTensorStringFill,
    IDEEPFallbackOp<GivenTensorFillOp<std::string, CPUContext>, SkipIndices<0>>);
REGISTER_IDEEP_OPERATOR(
    MSRAFill,
    IDEEPFallbackOp<MSRAFillOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    XavierFill,
    IDEEPFallbackOp<XavierFillOp<float, CPUContext>>);

// Loss operators
REGISTER_IDEEP_OPERATOR(
    Accuracy,
    IDEEPFallbackOp<AccuracyOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    AveragedLoss,
    IDEEPFallbackOp<AveragedLoss<float, CPUContext>, SkipIndices<0>>);
REGISTER_IDEEP_OPERATOR(
    AveragedLossGradient,
    IDEEPFallbackOp<AveragedLossGradient<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    LabelCrossEntropy,
    IDEEPFallbackOp<LabelCrossEntropyOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    LabelCrossEntropyGradient,
    IDEEPFallbackOp<LabelCrossEntropyGradientOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    L1Distance,
    IDEEPFallbackOp<L1DistanceOp<float, CPUContext>>);
// Softmax operators
REGISTER_IDEEP_OPERATOR(
    Softmax,
    IDEEPFallbackOp<SoftmaxOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    SoftmaxGradient,
    IDEEPFallbackOp<SoftmaxGradientOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    SoftmaxWithLoss,
    IDEEPFallbackOp<SoftmaxWithLossOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    CTCBeamSearchDecoder,
    IDEEPFallbackOp<CTCBeamSearchDecoderOp<CPUContext>>);

REGISTER_IDEEP_OPERATOR(
    AddGradient,
    IDEEPFallbackOp<BinaryElementwiseGradientOp<
      NumericTypes, CPUContext, AddFunctor<CPUContext>>>);
REGISTER_IDEEP_OPERATOR(
    MulGradient,
    IDEEPFallbackOp<BinaryElementwiseGradientOp<
      NumericTypes, CPUContext, MulFunctor<CPUContext>>>);
REGISTER_IDEEP_OPERATOR(
    StopGradient,
    IDEEPFallbackOp<StopGradientOp<CPUContext>>);
// Training operators
REGISTER_IDEEP_OPERATOR(
    Iter,
    IDEEPFallbackOp<IterOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    LearningRate,
    IDEEPFallbackOp<LearningRateOp<float, CPUContext>>);

REGISTER_IDEEP_OPERATOR(
    AffineChannel,
    IDEEPFallbackOp<AffineChannelOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    CollectAndDistributeFpnRpnProposals,
    IDEEPFallbackOp<CollectAndDistributeFpnRpnProposalsOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    BBoxTransform,
    IDEEPFallbackOp<BBoxTransformOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    BoxWithNMSLimit,
    IDEEPFallbackOp<BoxWithNMSLimitOp<CPUContext>, SkipIndices<0,1,2>>);
REGISTER_IDEEP_OPERATOR(
    GenerateProposals,
    IDEEPFallbackOp<GenerateProposalsOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    ConvTransposeGradient,
    IDEEPFallbackOp<ConvTransposeGradientOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    MulGradient,
    IDEEPFallbackOp<BinaryElementwiseGradientOp<
        NumericTypes,
        CPUContext,
        MulFunctor<CPUContext>>>);
REGISTER_IDEEP_OPERATOR(Adam, IDEEPFallbackOp<AdamOp<float, CPUContext>>);

} // namespace caffe2
