#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;  // NOLINT(build/namespaces)
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;


/********************/
/* helper functions */
/********************/

int ReadGPUIntegerScalar(const int* tensor_data);
void Zero1DKernelLauncher(int* input, int N);
void Zero2DKernelLauncher(int* input, int N, int M);
void Zero3DKernelLauncher(float* input, int N, int M, int K);


/************************/
/* n_rf_events function */
/************************/

REGISTER_OP("NRfEvents")
    .Input("rf_idx: int32")
    .Input("lengths: int32")
    .Attr("num_rf: int")
    .Output("rf_events_count: int32")
    .SetShapeFn([](InferenceContext* c) {
        // Retrieve the num_rf attribute
        int num_rf;
        TF_RETURN_IF_ERROR(c->GetAttr("num_rf", &num_rf));
        // Retrieve the input batch size
        auto input_shape = c->input(0);
        auto batch_size = c->Dim(input_shape, 0);

        c->set_output(0, c->MakeShape({batch_size, num_rf}));
        return Status::OK();
    });

void NRfEventsKernelLauncher(const int* rf_idx, const int* lengths,
                             int batch_size, int event_size,
                             int num_rf, int* rf_events_count);

class NRfEventsOp : public OpKernel {

	private:
		int num_rf_;

	public:
		explicit NRfEventsOp(OpKernelConstruction* context) : OpKernel(context) {
			// Grab the attributes (aka, method's inputs that are fixed
            // after graph construction)
            OP_REQUIRES_OK(context, context->GetAttr("num_rf", &num_rf_));
		}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& rf_idx_tensor = context->input(0);
            auto rf_idx = rf_idx_tensor.flat<int32>();
            const Tensor& lengths_tensor = context->input(1);
            auto lengths = lengths_tensor.flat<int32>();
            // Retrieve the input tensor shapes
            auto shape_vec = rf_idx_tensor.shape();
            auto batch_size = shape_vec.dim_size(0);
            auto event_size = shape_vec.dim_size(1);

            // Create an output tensor
            Tensor* rf_events_count_tensor = nullptr;
            OP_REQUIRES_OK(
                context, context->allocate_output(0, TensorShape({batch_size, num_rf_}),
                                                  &rf_events_count_tensor));
            auto rf_events_count = rf_events_count_tensor->template flat<int32>();
            // Set all the elements of the output tensor to 0
            Zero2DKernelLauncher(rf_events_count.data(), batch_size, num_rf_);

            // Launch the kernel
            NRfEventsKernelLauncher(rf_idx.data(),
                                    lengths.data(),
                                    batch_size,
                                    event_size,
                                    num_rf_,
                                    rf_events_count.data());
        }
};

REGISTER_KERNEL_BUILDER(Name("NRfEvents").Device(DEVICE_GPU), NRfEventsOp);


/*****************************/
/* group_rf_bounded function */
/*****************************/

REGISTER_OP("GroupRfBounded")
    .Input("features: float32")
    .Input("rf_ids: int32")
    .Input("lengths: int32")
    .Input("rf_offsets: int32")
    .Input("rf_counts: int32")
    .Input("new_event_size: int32")
    .Input("new_batch_size: int32")
    .Input("max_rf_events: int32")
    .Attr("h: int")
    .Attr("w: int")
    .Output("gr_batch_id: int32")
    .Output("gr_last_id: int32")
    .Output("gr_h: int32")
    .Output("gr_w: int32")
    .Output("groups: float32")
    .SetShapeFn([](InferenceContext* c) {
        auto feature_size = c->Dim(c->input(0), 2);

        ShapeHandle new_event_size_shape, new_batch_size_shape, groups_shape;
        TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(5, &new_event_size_shape));
        TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(6, &new_batch_size_shape));
        TF_RETURN_IF_ERROR(c->Concatenate(new_event_size_shape,
                                          new_batch_size_shape,
                                          &groups_shape));
        TF_RETURN_IF_ERROR(c->Concatenate(groups_shape,
                                          c->MakeShape({feature_size}),
                                          &groups_shape));

        c->set_output(0, new_batch_size_shape);
        c->set_output(1, new_batch_size_shape);
        c->set_output(2, new_batch_size_shape);
        c->set_output(3, new_batch_size_shape);
        c->set_output(4, groups_shape);

        return Status::OK();
    });

void GroupRfBoundedKernelLauncher(const float* features, const int* rf_ids, const int* lengths,
                                  const int* rf_offsets, const int* rf_counts, float* groups,
                                  int* gr_last_id, int* gr_batch_id, int* gr_h, int* gr_w,
                                  const int w, const int h, const int batch_size,
                                  const int event_size, const int feature_size,
                                  const int new_batch_size, const int max_rf_events);

class GroupRfBoundedOp : public OpKernel {

	private:
		int h_, w_;

	public:
		explicit GroupRfBoundedOp(OpKernelConstruction* context) : OpKernel(context) {
			// Grab the attributes (aka, method's inputs that are fixed
            // after graph construction)
            OP_REQUIRES_OK(context, context->GetAttr("h", &h_));
            OP_REQUIRES_OK(context, context->GetAttr("w", &w_));
		}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& features_tensor = context->input(0);
            auto features = features_tensor.flat<float>();
            const Tensor& rf_ids_tensor = context->input(1);
            auto rf_ids = rf_ids_tensor.flat<int32>();
            const Tensor& lengths_tensor = context->input(2);
            auto lengths = lengths_tensor.flat<int32>();
            const Tensor& rf_offsets_tensor = context->input(3);
            auto rf_offsets = rf_offsets_tensor.flat<int32>();
            const Tensor& rf_counts_tensor = context->input(4);
            auto rf_counts = rf_counts_tensor.flat<int32>();
            const Tensor& new_event_size_tensor = context->input(5);
            const Tensor& new_batch_size_tensor = context->input(6);
            const Tensor& max_rf_events_tensor = context->input(7);

            OP_REQUIRES(context, new_event_size_tensor.dims() == 1,
                errors::InvalidArgument("new_event_size must be 1-D",
                    new_event_size_tensor.shape().DebugString()));
            OP_REQUIRES(context, new_event_size_tensor.dim_size(0) == 1,
                errors::InvalidArgument("new_event_size must have 1 element",
                    new_event_size_tensor.shape().DebugString()));

            OP_REQUIRES(context, new_batch_size_tensor.dims() == 1,
                errors::InvalidArgument("new_batch_size must be 1-D",
                    new_batch_size_tensor.shape().DebugString()));
            OP_REQUIRES(context, new_batch_size_tensor.dim_size(0) == 1,
                errors::InvalidArgument("new_batch_size must have 1 element",
                    new_batch_size_tensor.shape().DebugString()));

            OP_REQUIRES(context, max_rf_events_tensor.dims() == 1,
                errors::InvalidArgument("max_rf_events must be 1-D",
                max_rf_events_tensor.shape().DebugString()));
            OP_REQUIRES(context, max_rf_events_tensor.dim_size(0) == 1,
                errors::InvalidArgument("max_rf_events must have 1 element",
                    max_rf_events_tensor.shape().DebugString()));

            int new_event_size = ReadGPUIntegerScalar(
                                    new_event_size_tensor.flat<int32>().data());
            int new_batch_size = ReadGPUIntegerScalar(
                                    new_batch_size_tensor.flat<int32>().data());
            int max_rf_events = ReadGPUIntegerScalar(
                                    max_rf_events_tensor.flat<int32>().data());

            // Retrieve the input tensor shapes
            const auto shape_vec = features_tensor.shape();
            const auto batch_size = shape_vec.dim_size(0);
            const auto event_size = shape_vec.dim_size(1);
            const auto feature_size = shape_vec.dim_size(2);

            // Create output tensors
            Tensor* gr_batch_id_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,
                                                TensorShape({new_batch_size}),
                                                &gr_batch_id_tensor));
            auto gr_batch_id = gr_batch_id_tensor->template flat<int32>();
            Tensor* gr_last_id_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1,
                                                TensorShape({new_batch_size}),
                                                &gr_last_id_tensor));
            auto gr_last_id = gr_last_id_tensor->template flat<int32>();
            Tensor* gr_h_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(2,
                                                TensorShape({new_batch_size}),
                                                &gr_h_tensor));
            auto gr_h = gr_h_tensor->template flat<int32>();
            Tensor* gr_w_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(3,
                                                TensorShape({new_batch_size}),
                                                &gr_w_tensor));
            auto gr_w = gr_w_tensor->template flat<int32>();
            Tensor* groups_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(4,
                                                TensorShape({new_event_size,
                                                             new_batch_size,
                                                             feature_size}),
                                                &groups_tensor));
            auto groups = groups_tensor->template flat<float>();

            // Set all the elements of the output tensors to 0
            Zero1DKernelLauncher(gr_batch_id.data(), new_batch_size);
            Zero1DKernelLauncher(gr_last_id.data(), new_batch_size);
            Zero1DKernelLauncher(gr_h.data(), new_batch_size);
            Zero1DKernelLauncher(gr_w.data(), new_batch_size);
            Zero3DKernelLauncher(groups.data(), new_event_size,
                                                new_batch_size,
                                                feature_size);

            // Launch the kernel
            GroupRfBoundedKernelLauncher(features.data(), rf_ids.data(),
                                         lengths.data(), rf_offsets.data(),
                                         rf_counts.data(), groups.data(),
                                         gr_last_id.data(), gr_batch_id.data(),
                                         gr_h.data(), gr_w.data(),
                                         w_, h_, batch_size, event_size,
                                         feature_size, new_batch_size,
                                         max_rf_events);
        }
};

REGISTER_KERNEL_BUILDER(Name("GroupRfBounded").Device(DEVICE_GPU), GroupRfBoundedOp);


/***************************/
/* sparse_adj_idx function */
/***************************/

REGISTER_OP("SparseAdjIdx")
    .Input("rf_ev_step: int32")
    .Input("rf_offsets: int32")
    .Input("rf_lens: int32")
    .Input("out_num_idx: int32")
    .Attr("max_rf_len: int")
    .Output("sparse_idx: int32")
    .SetShapeFn([](InferenceContext* c) {
        // Retrieve the max_rf_len attribute
        int max_rf_len;
        TF_RETURN_IF_ERROR(c->GetAttr("max_rf_len", &max_rf_len));

        // The output has shape [out_num_idx, 3]
        ShapeHandle out_shape;
        TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(3, &out_shape));
        TF_RETURN_IF_ERROR(c->Concatenate(out_shape, c->MakeShape({3}),
                                          &out_shape));
        c->set_output(0, out_shape);

        return Status::OK();
    });

void SparseAdjIdxKernelLauncher(const int* rf_ev_step, const int* rf_offsets,
                                const int* rf_lens, const int num_rf,
                                const int max_rf_len, int* sparse_idx);

class SparseAdjIdxOp : public OpKernel {

	private:
		int max_rf_len_;

	public:
		explicit SparseAdjIdxOp(OpKernelConstruction* context) : OpKernel(context) {
			// Grab the attributes (aka, method's inputs that are fixed
            // after graph construction)
            OP_REQUIRES_OK(context, context->GetAttr("max_rf_len", &max_rf_len_));
		}

		void Compute(OpKernelContext* context) override {
		    // Grab the input tensors
            const Tensor& rf_ev_step_tensor = context->input(0);
            auto rf_ev_step = rf_ev_step_tensor.flat<int>();
            const Tensor& rf_offsets_tensor = context->input(1);
            auto rf_offsets = rf_offsets_tensor.flat<int>();
            const Tensor& rf_lens_tensor = context->input(2);
            auto rf_lens = rf_lens_tensor.flat<int>();
            const Tensor& out_num_idx_tensor = context->input(3);

            OP_REQUIRES(context, out_num_idx_tensor.dims() == 1,
                errors::InvalidArgument("out_num_idx_tensor must be 1-D",
                    out_num_idx_tensor.shape().DebugString()));
            OP_REQUIRES(context, out_num_idx_tensor.dim_size(0) == 1,
                errors::InvalidArgument("out_num_idx_tensor must have 1 element",
                    out_num_idx_tensor.shape().DebugString()));

            auto shape_vec = rf_ev_step_tensor.shape();
            auto num_rf = shape_vec.dim_size(0);
            auto out_num_idx_cpu = ReadGPUIntegerScalar(
                                     out_num_idx_tensor.flat<int32>().data());

            // Create output tensors
            Tensor* sparse_idx_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,
                                                TensorShape({out_num_idx_cpu, 3}),
                                                &sparse_idx_tensor));
            auto sparse_idx = sparse_idx_tensor->template flat<int32>();

            // Launch the kernel
            SparseAdjIdxKernelLauncher(rf_ev_step.data(),
                                       rf_offsets.data(),
                                       rf_lens.data(),
                                       num_rf,
                                       max_rf_len_,
                                       sparse_idx.data());
		}
};

REGISTER_KERNEL_BUILDER(Name("SparseAdjIdx").Device(DEVICE_GPU), SparseAdjIdxOp);


/**************************/
/* get_centroids function */
/**************************/

REGISTER_OP("GetCentroids")
    .Input("events: float32")
    .Input("rf_ev_step: int32")
    .Attr("max_rf_len: int")
    .Output("centroids: float32")
    .SetShapeFn([](InferenceContext* c) {
        // Retrieve the max_rf_len attribute
        int max_rf_len;
        TF_RETURN_IF_ERROR(c->GetAttr("max_rf_len", &max_rf_len));

        // Retrieve the input sizes
        // events.shape = [num_rf, max_padded_time, feature_size]
        auto input_shape = c->input(0);
        auto num_rf = c->Dim(input_shape, 0);
        auto feature_size = c->Dim(input_shape, 2);

        c->set_output(0, c->MakeShape({num_rf, max_rf_len, feature_size}));
        return Status::OK();
    });

void GetCentroidsKernelLauncher(const float* events, const int* rf_ev_step,
                                const int max_rf_len, const int num_rf,
                                const int event_size, const int feature_size,
                                float* centroids);

class GetCentroidsOp : public OpKernel {

	private:
		int max_rf_len_;

	public:
		explicit GetCentroidsOp(OpKernelConstruction* context) : OpKernel(context) {
			// Grab the attributes (aka, method's inputs that are fixed
            // after graph construction)
            OP_REQUIRES_OK(context, context->GetAttr("max_rf_len", &max_rf_len_));
		}

		void Compute(OpKernelContext* context) override {
		    // Grab the input tensors
            const Tensor& events_tensor = context->input(0);
            auto events = events_tensor.flat<float>();
            const Tensor& rf_ev_step_tensor = context->input(1);
            auto rf_ev_step = rf_ev_step_tensor.flat<int>();

            auto shape_vec = events_tensor.shape();
            auto num_rf = shape_vec.dim_size(0);
            auto event_size = shape_vec.dim_size(1);
            auto feature_size = shape_vec.dim_size(2);

            // Create output tensors
            Tensor* centroids_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,
                                                TensorShape({num_rf,
                                                             max_rf_len_,
                                                             feature_size}),
                                                &centroids_tensor));
            auto centroids = centroids_tensor->template flat<float>();

            // Launch the kernel
            GetCentroidsKernelLauncher(events.data(),
                                       rf_ev_step.data(),
                                       max_rf_len_,
                                       num_rf,
                                       event_size,
                                       feature_size,
                                       centroids.data());
		}
};

REGISTER_KERNEL_BUILDER(Name("GetCentroids").Device(DEVICE_GPU), GetCentroidsOp);


/******************************/
/* n_interval_events function */
/******************************/

REGISTER_OP("NIntervalEvents")
    .Input("ts_percent: float32")
    .Input("lengths: int32")
    .Attr("n_intervals: int")
    .Output("interval_events_count: int32")
    .SetShapeFn([](InferenceContext* c) {
        // Retrieve the num_rf attribute
        int n_intervals;
        TF_RETURN_IF_ERROR(c->GetAttr("n_intervals", &n_intervals));
        // Retrieve the input batch size
        auto input_shape = c->input(0);
        auto batch_size = c->Dim(input_shape, 0);

        c->set_output(0, c->MakeShape({batch_size, n_intervals}));
        return Status::OK();
    });

void NIntervalEventsKernelLauncher(const float* ts_percent, const int* lengths,
                                   int batch_size, int event_size, int n_intervals,
                                   int* interval_events_count);

class NIntervalEventsOp : public OpKernel {

	private:
		int n_intervals_;

	public:
		explicit NIntervalEventsOp(OpKernelConstruction* context) : OpKernel(context) {
			// Grab the attributes (aka, method's inputs that are fixed
            // after graph construction)
            OP_REQUIRES_OK(context, context->GetAttr("n_intervals", &n_intervals_));
		}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& ts_percent_tensor = context->input(0);
            auto ts_percent = ts_percent_tensor.flat<float>();
            const Tensor& lengths_tensor = context->input(1);
            auto lengths = lengths_tensor.flat<int32>();
            // Retrieve the input tensor shapes
            auto shape_vec = ts_percent_tensor.shape();
            auto batch_size = shape_vec.dim_size(0);
            auto event_size = shape_vec.dim_size(1);

            // Create an output tensor
            Tensor* interval_events_count_tensor = nullptr;
            OP_REQUIRES_OK(
                context, context->allocate_output(0, TensorShape({batch_size, n_intervals_}),
                                                  &interval_events_count_tensor));
            auto interval_events_count = interval_events_count_tensor->template flat<int32>();
            // Set all the elements of the output tensor to 0
            Zero2DKernelLauncher(interval_events_count.data(), batch_size, n_intervals_);

            // Launch the kernel
            NIntervalEventsKernelLauncher(ts_percent.data(),
                                          lengths.data(),
                                          batch_size,
                                          event_size,
                                          n_intervals_,
                                          interval_events_count.data());
        }
};

REGISTER_KERNEL_BUILDER(Name("NIntervalEvents").Device(DEVICE_GPU), NIntervalEventsOp);


/*******************************/
/* intervals_to_batch function */
/*******************************/

REGISTER_OP("IntervalsToBatch")
    .Input("events: float32")
    .Input("n_interval_events: int32")
    .Input("new_event_size: int32")
    .Output("interval_events: float32")
    .SetShapeFn([](InferenceContext* c) {
        // Retrieve the input batch size
        auto events_shape = c->input(0);
        auto batch_size = c->Dim(events_shape, 0);
        auto features_size = c->Dim(events_shape, 2);
        auto intervals_shape = c->input(1);
        auto n_intervals = c->Dim(intervals_shape, 1);

        DimensionHandle new_batch_size_dim;
        ShapeHandle new_event_size_shape, new_events_shape;
        TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(2, &new_event_size_shape));
        TF_RETURN_IF_ERROR(c->Multiply(n_intervals, batch_size,
                                       &new_batch_size_dim));
        TF_RETURN_IF_ERROR(c->Concatenate(c->MakeShape({new_batch_size_dim}),
                                          new_event_size_shape,
                                          &new_events_shape));
        TF_RETURN_IF_ERROR(c->Concatenate(new_events_shape,
                                          c->MakeShape({features_size}),
                                          &new_events_shape));
        // shape: [batch_size * n_intervals, new_event_size, features_size]
        c->set_output(0, new_events_shape);
        return Status::OK();
    });

void IntervalsToBatchKernelLauncher(const float* events, const int* n_interval_events,
                                    int batch_size, int event_size, int feature_size,
                                    int n_intervals,  int new_event_size, float* new_events);

class IntervalsToBatchOp : public OpKernel {

	public:
		explicit IntervalsToBatchOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& events_tensor = context->input(0);
            auto events = events_tensor.flat<float>();
            const Tensor& n_interval_events_tensor = context->input(1);
            auto n_interval_events = n_interval_events_tensor.flat<int32>();
            const Tensor& new_event_size_tensor = context->input(2);
            // Retrieve the input tensor shapes
            int new_event_size = ReadGPUIntegerScalar(
                                    new_event_size_tensor.flat<int32>().data());
            auto events_shape = events_tensor.shape();
            auto batch_size = events_shape.dim_size(0);
            auto event_size = events_shape.dim_size(1);
            auto feature_size = events_shape.dim_size(2);
            auto intervals_shape = n_interval_events_tensor.shape();
            auto n_intervals = intervals_shape.dim_size(1);

            // Create an output tensor
            Tensor* new_events_tensor = nullptr;
            OP_REQUIRES_OK(
                context, context->allocate_output(0, TensorShape({batch_size * n_intervals,
                                                                  new_event_size,
                                                                  feature_size}),
                                                  &new_events_tensor));
            auto new_events = new_events_tensor->template flat<float>();
            // Set all the elements of the output tensor to 0
            Zero3DKernelLauncher(new_events.data(),
                                 batch_size * n_intervals,
                                 new_event_size,
                                 feature_size);

            // Launch the kernel
            IntervalsToBatchKernelLauncher(events.data(),
                                           n_interval_events.data(),
                                           batch_size,
                                           event_size,
                                           feature_size,
                                           n_intervals,
                                           new_event_size,
                                           new_events.data());
        }
};

REGISTER_KERNEL_BUILDER(Name("IntervalsToBatch").Device(DEVICE_GPU), IntervalsToBatchOp);