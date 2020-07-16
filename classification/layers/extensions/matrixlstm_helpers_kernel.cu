#include <torch/extension.h>
#include <vector>
#include <stdio.h>

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>



/**************/
/* max_scalar */
/**************/

template <typename scalar_t>
__global__ void min2d_scalar_kernel(const scalar_t* __restrict__ input,
                                    const scalar_t scalar,
                                    scalar_t* __restrict__ output,
                                    int64_t N,
                                    int64_t M) {

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (n < N & m < M){
        auto val = input[n*M + m];
        if (val > scalar)
            output[n*M + m] = scalar;
        else
            output[n*M + m] = val;
    }
}

torch::Tensor min2d_scalar_cuda(torch::Tensor input, torch::Tensor scalar){

    // input.shape = [N, M]
	const auto N = input.size(0);
	const auto M = input.size(1);

    // Create the output tensor
    auto output = torch::zeros_like(input);

    // Split the first dimension over threadsPerBlock.x threads
    // and the second dimension over threadsPerBlock.y threads
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((int)((N + threadsPerBlock.x - 1) / threadsPerBlock.x),
                   (int)((M + threadsPerBlock.y - 1) / threadsPerBlock.y));

    AT_DISPATCH_INTEGRAL_TYPES(input.type(), "n_rf_events_cuda", ([&] {
		min2d_scalar_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
			input.data<scalar_t>(),
			scalar.cpu().item().to<scalar_t>(),
			output.data<scalar_t>(),
			N,
			M);
	}));

    return output;
}


/************************/
/* n_rf_events function */
/************************/

template <typename scalar_t>
__global__ void n_rf_events_cuda_kernel(const scalar_t* __restrict__ rf_idx,
                                        const scalar_t* __restrict__ lengths,
										scalar_t* __restrict__ rf_events_count,
										const int64_t batch_size,
										const int64_t event_size,
										const int64_t num_rf) {

    const int64_t batch_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_id < batch_size){
        for (int64_t e = 0; e < event_size & e < lengths[batch_id]; e++){
            int64_t read_index = batch_id * event_size + e;
            // rf_idx[batch_id][e]
            int64_t rf_id = rf_idx[read_index];
            if (rf_id == -1)
                continue;

            int64_t write_index = batch_id * num_rf + rf_id;
            // rf_events_count[batch_id][rf_id]
            rf_events_count[write_index] += 1;
        }
    }
}


torch::Tensor n_rf_events_cuda(torch::Tensor rf_idx, torch::Tensor lengths, int num_rf){

	// rf_idx.shape = [batch_size, n_events]
	const auto batch_size = rf_idx.size(0);
	const auto event_size = rf_idx.size(1);

	// Create a tensor to hold output counts
	auto options = torch::TensorOptions(at::kLong).device(at::kCUDA);
	auto rf_events_count = torch::zeros({batch_size, num_rf}, options);

    // Allocate a thread for each sample, each one will process all the events
    // in the sample (just the batch loop is parallelized)
	int threadsPerBlock = 32;
	int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

	AT_DISPATCH_INTEGRAL_TYPES(rf_idx.type(), "n_rf_events_cuda", ([&] {
		n_rf_events_cuda_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
			rf_idx.data<scalar_t>(),
			lengths.data<scalar_t>(),
			rf_events_count.data<scalar_t>(),
			batch_size,
			event_size,
			num_rf);
	}));

	return rf_events_count;
}


/********************************/
/* n_rf_events_overlap function */
/********************************/

template <typename scalar_t>
__global__ void n_rf_events_overlap_cuda_kernel(const scalar_t* __restrict__ pix_idx,
                                                const scalar_t* __restrict__ lengths,
                                                const scalar_t* __restrict__ pix2rf,
                                                scalar_t* __restrict__ rf_events_count,
                                                const int64_t batch_size,
                                                const int64_t event_size,
                                                const int64_t num_rf,
                                                const int64_t max_overlap) {

    const int64_t batch_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_id < batch_size){
        for (int64_t e = 0; e < event_size & e < lengths[batch_id]; e++){
            int64_t read_index = batch_id * event_size + e;
            // pix_idx[batch_id][e]
            int64_t pix_id = pix_idx[read_index];
            for (int64_t r = 0; r < max_overlap; r++){
                // pix2rf[pix_id][r]
                int64_t rf_id = pix2rf[pix_id * max_overlap + r];
                if (rf_id < 0) break;

                int64_t write_index = batch_id * num_rf + rf_id;
                // rf_events_count[batch_id][rf_id]
                rf_events_count[write_index] += 1;
            }
        }
    }
}


torch::Tensor n_rf_events_overlap_cuda(torch::Tensor pix_idx,
                                       torch::Tensor lengths,
                                       torch::Tensor pix2rf,
                                       int num_rf){

	// pix_idx.shape = [batch_size, n_events]
	const auto batch_size = pix_idx.size(0);
	const auto event_size = pix_idx.size(1);
	const auto max_overlap = pix2rf.size(1);

	// Create a tensor to hold output counts
	auto options = torch::TensorOptions(at::kLong).device(at::kCUDA);
	auto rf_events_count = torch::zeros({batch_size, num_rf}, options);

    // Allocate a thread for each sample, each one will process all the events
    // in the sample (just the batch loop is parallelized)
	int threadsPerBlock = 32;
	int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

	AT_DISPATCH_INTEGRAL_TYPES(pix_idx.type(), "n_rf_events_cuda", ([&] {
		n_rf_events_overlap_cuda_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
			pix_idx.data<scalar_t>(),
			lengths.data<scalar_t>(),
			pix2rf.data<scalar_t>(),
			rf_events_count.data<scalar_t>(),
			batch_size,
			event_size,
			num_rf,
			max_overlap);
	}));

	return rf_events_count;
}


/*********************/
/* group_rf function */
/*********************/

template <typename scalar_t>
__global__ void group_rf_cuda_kernel(const scalar_t* __restrict__ features,
                                     const int64_t* __restrict__ rf_ids,
			                         const int64_t* __restrict__ lengths,
			                         const int64_t* __restrict__ rf_offsets,
			                         scalar_t* __restrict__ groups,
			                         int64_t* __restrict__ gr_last_id,
                                     int64_t* __restrict__ gr_batch_id,
                                     int64_t* __restrict__ gr_h,
                                     int64_t* __restrict__ gr_w,
                                     const int64_t out_w,
			                         const int64_t batch_size,
			                         const int64_t event_size,
			                         const int64_t feature_size,
			                         const int64_t num_rf,
			                         const int64_t new_batch_size){

    const int64_t batch_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_id < batch_size){
        for (int64_t e = 0; e < event_size & e < lengths[batch_id]; e++){
            // rf_idx[batch_id][e]
            const int64_t rf_id = rf_ids[batch_id * event_size + e];
            if (rf_id == -1)
                continue;
            // Retrieve the position of the receptive field within the output tensor
            // rf_offsets[batch_id][rf_id]
            const int64_t rf_pos = rf_offsets[batch_id * num_rf + rf_id] - 1;
            // Retrieve the number of events already placed inside the target rf
            // const int64_t event_pos = rf_event_counts[rf_id];
            const int64_t event_pos = gr_last_id[rf_pos];

            // groups[event_pos][rf_pos][:] = features[batch_id][e][:]
            const int64_t write_offset = event_pos * (new_batch_size * feature_size) \
                                         + rf_pos * feature_size;
            const int64_t read_offset = batch_id * (event_size * feature_size) \
                                        + e * feature_size;
            for (int64_t f = 0; f < feature_size; f++)
                groups[write_offset + f] = features[read_offset + f];

            // Increment the number of events in the current receptive field
            gr_last_id[rf_pos] += 1;

            // We write receptive field information just one time (it is the same for
            // all the events in the same receptive field), ie, the first time we process
            // an event inside the receptive field
            if (event_pos == 0){
                gr_batch_id[rf_pos] = batch_id;
                gr_h[rf_pos] = (int)(rf_id / out_w);
                gr_w[rf_pos] = (int)(rf_id % out_w);
            }
        }
    }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
group_rf_cuda(torch::Tensor features, torch::Tensor rf_ids,
              torch::Tensor lengths, int h, int w){

    const auto batch_size = features.size(0);
    const auto event_size = features.size(1);
    const auto feature_size = features.size(2);

    // Compute the number of events in each receptive field of every sample
    const auto rf_counts = n_rf_events_cuda(rf_ids, lengths, h*w);
    // Compute the new event length (max events in a receptive field)
    const auto new_event_size = torch::max(rf_counts).cpu().item().to<int64_t>();
    // Compute the position that each non empty receptive field will have in the output tensor
    const auto rf_offsets = torch::cumsum(rf_counts.gt(0).view({-1}), /*dim=*/-1).view(rf_counts.sizes());
    // Compute the new flat batch size (tot num of non empty receptive fields)
    const auto new_batch_size = rf_offsets[batch_size-1][h*w-1].cpu().item().to<int64_t>();

    // Create a tensor to hold output groups
	auto groups = torch::zeros({new_event_size, new_batch_size, feature_size}, features.options());
	auto gr_last_id = torch::zeros({new_batch_size}, features.options().dtype(at::kLong));
	auto gr_batch_id = torch::zeros({new_batch_size}, features.options().dtype(at::kLong));
	auto gr_h = torch::zeros({new_batch_size}, features.options().dtype(at::kLong));
	auto gr_w = torch::zeros({new_batch_size}, features.options().dtype(at::kLong));

	// Allocate a thread for each sample, each one will process all the events
    // in the sample (just the batch loop is parallelized)
	int threadsPerBlock = 32;
	int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

	AT_DISPATCH_ALL_TYPES(features.type(), "group_rf_cuda", ([&] {
		group_rf_cuda_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
			features.data<scalar_t>(),
			rf_ids.data<int64_t>(),
			lengths.data<int64_t>(),
			rf_offsets.data<int64_t>(),
			groups.data<scalar_t>(),
			gr_last_id.data<int64_t>(),
			gr_batch_id.data<int64_t>(),
			gr_h.data<int64_t>(),
			gr_w.data<int64_t>(),
			w,
			batch_size,
			event_size,
			feature_size,
			h*w,
			new_batch_size);
	}));

	// Decrement all values by one (we want the id, not the count)
	gr_last_id = gr_last_id.sub(1);

    return std::make_tuple(gr_batch_id, gr_last_id, gr_h, gr_w, groups);
}


/*****************************/
/* group_rf_bounded function */
/*****************************/

template <typename scalar_t>
__global__ void group_rf_bounded_cuda_kernel(const scalar_t* __restrict__ features,
                                             const int64_t* __restrict__ rf_ids,
                                             const int64_t* __restrict__ lengths,
                                             const int64_t* __restrict__ rf_offsets,
                                             const int64_t* __restrict__ rf_lengths,
                                             scalar_t* __restrict__ groups,
                                             int64_t* __restrict__ gr_last_id,
                                             int64_t* __restrict__ gr_batch_id,
                                             int64_t* __restrict__ gr_h,
                                             int64_t* __restrict__ gr_w,
                                             const int64_t out_w,
                                             const int64_t batch_size,
                                             const int64_t event_size,
                                             const int64_t feature_size,
                                             const int64_t num_rf,
                                             const int64_t new_batch_size,
                                             const int64_t max_rf_events){

    const int64_t batch_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_rf_finished = 0;

    if (batch_id < batch_size){
        for (int64_t e = lengths[batch_id] - 1; e >= 0 & num_rf_finished < num_rf; e--){
            // rf_idx[batch_id][e]
            const int64_t rf_id = rf_ids[batch_id * event_size + e];
            if (rf_id == -1)
                continue;
            // rf_lengths[batch_id][rf_id]
            const int64_t rf_len = rf_lengths[batch_id * num_rf + rf_id];
            // We may be asked not to take all the events in the receptive field, but only
            // a limited number. In that case (max_rf_events < rf_len) we take an event at regular
            // intervals to cover the whole receptive field temporal window.
            int ev_step = 1;
            long last_write_pos = rf_len - 1;
            if (max_rf_events < rf_len){
                ev_step = rf_len / (max_rf_events - 1);
                last_write_pos = max_rf_events - 1;
            }
            // Retrieve the position of the receptive field within the output tensor
            // rf_offsets[batch_id][rf_id]
            const int64_t rf_pos = rf_offsets[batch_id * num_rf + rf_id] - 1;

            if (gr_last_id[rf_pos] < rf_len){
                // Retrieve the number of events already placed inside the target rf
                // const int64_t event_pos = rf_event_counts[rf_id];
                const int64_t event_pos = rf_len - 1 - gr_last_id[rf_pos];

                // By dividing event_pos by ev_step we override the same position multiple
                // times, stepping to the next position every ev_step. Note: casting to
                // integer make write_event_pos change only change every ev_step values
                int64_t write_event_pos = last_write_pos - (long)(gr_last_id[rf_pos] / ev_step);
                const int64_t prev_write_event_pos = last_write_pos - \
                                                     (long)(max(0L, (long)(gr_last_id[rf_pos]) - 1) / ev_step);

                // If the position has not changed (we didn't move by at least ev_step positions
                // wrt the previous step) we skip the write operation. Note: we always write the
                // first and last event in the receptive field!
                if (gr_last_id[rf_pos] + 1 == rf_len)
                    write_event_pos = 0;
                if (write_event_pos >= 0 && (write_event_pos != prev_write_event_pos ||
                                            gr_last_id[rf_pos] == 0 ||
                                            gr_last_id[rf_pos] + 1 == rf_len)){
                    // groups[write_event_pos][rf_pos][:] = features[batch_id][e][:]
                    const int64_t write_offset = write_event_pos * (new_batch_size * feature_size) \
                                                 + rf_pos * feature_size;
                    const int64_t read_offset = batch_id * (event_size * feature_size) \
                                                + e * feature_size;
                    for (int64_t f = 0; f < feature_size; f++)
                        groups[write_offset + f] = features[read_offset + f];
                }

                // Increment the number of events in the current receptive field
                // Note: if ev_step > 1 this will count the number of read events
                // since we actually only write every ev_step. We will divide the
                // total number at the end
                gr_last_id[rf_pos] += 1;
                // Check if the current one was the last event in the current receptive field
                // in this case increment the number of finished receptive fields
                if (gr_last_id[rf_pos] == rf_len)
                    num_rf_finished += 1;

                // We write receptive field information just one time (it is the same for
                // all the events in the same receptive field), ie, the first time we process
                // an event inside the receptive field
                if (event_pos == 0){
                    gr_batch_id[rf_pos] = batch_id;
                    gr_h[rf_pos] = (int)(rf_id / out_w);
                    gr_w[rf_pos] = (int)(rf_id % out_w);
                }
            }
        }

        // We incremented gr_last_id each time even if a write was not actually
        // performed due to ev_step > 1. Here we fix gr_last_id replacing the
        // actual length for the receptive fields having more events
        for (int64_t rf_id = 0; rf_id < num_rf; rf_id++){
            const int64_t rf_pos = rf_offsets[batch_id * num_rf + rf_id] - 1;
            const int64_t rf_len = rf_lengths[batch_id * num_rf + rf_id];
            if (max_rf_events < rf_len)
                gr_last_id[rf_pos] = max_rf_events;
        }
    }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
group_rf_bounded_cuda(torch::Tensor features, torch::Tensor rf_ids,
                      torch::Tensor lengths, torch::Tensor max_rf_events,
                      int h, int w, bool keep_most_recent){

    const auto batch_size = features.size(0);
    const auto event_size = features.size(1);
    const auto feature_size = features.size(2);
    const auto max_rf_events_cpu = max_rf_events.cpu().item().to<int64_t>();

    // Compute the number of events in each receptive field of every sample
    auto rf_counts = n_rf_events_cuda(rf_ids, lengths, h*w);
    // Bound the number of events in each receptive field based on the provided max
    torch::Tensor  bounded_rf_counts;
    if (max_rf_events_cpu > 0)
        bounded_rf_counts = min2d_scalar_cuda(rf_counts, max_rf_events);
    else
        bounded_rf_counts = rf_counts;

    // Compute the new event length (max events in a receptive field)
    const auto new_event_size = torch::max(bounded_rf_counts).cpu().item().to<int64_t>();
    // Compute the position that each non empty receptive field will have in the output tensor
    const auto rf_offsets = torch::cumsum(bounded_rf_counts.gt(0).view({-1}), /*dim=*/-1).view(bounded_rf_counts.sizes());
    // Compute the new flat batch size (tot num of non empty receptive fields)
    const auto new_batch_size = rf_offsets[batch_size-1][h*w-1].cpu().item().to<int64_t>();

    // Create a tensor to hold output groups
	auto groups = torch::zeros({new_event_size, new_batch_size, feature_size}, features.options());
	auto gr_last_id = torch::zeros({new_batch_size}, features.options().dtype(at::kLong));
	auto gr_batch_id = torch::zeros({new_batch_size}, features.options().dtype(at::kLong));
	auto gr_h = torch::zeros({new_batch_size}, features.options().dtype(at::kLong));
	auto gr_w = torch::zeros({new_batch_size}, features.options().dtype(at::kLong));

	// Allocate a thread for each sample, each one will process all the events
    // in the sample (just the batch loop is parallelized)
	int threadsPerBlock = 32;
	int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

    // We have two strategies based on the value of keep_most_recent
    // - keep_most_recent is True: we pass to group_rf_bounded_cuda_kernel the
    //      rf_counts bounded to max_rf_events. This way the grouping algorithm
    //      will only look at the last max_rf_events (or less) events in each
    //      receptive field, copying each of them (step = 1) in the output
    // - keep_most_recent is False: we pass to group_rf_bounded_cuda_kernel the
    //      NOT bounded version of rf_counts so that the algorithm will look at
    //      all the events in the receptive field. However, it will write with
    //      a step > 1, not copying some of the events in the output tensor.
    //      We always compute new_event_size using the bounded version!
    if (keep_most_recent)
        rf_counts = bounded_rf_counts;

	AT_DISPATCH_ALL_TYPES(features.type(), "group_rf_cuda", ([&] {
		group_rf_bounded_cuda_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
			features.data<scalar_t>(),
			rf_ids.data<int64_t>(),
			lengths.data<int64_t>(),
			rf_offsets.data<int64_t>(),
			rf_counts.data<int64_t>(),
			groups.data<scalar_t>(),
			gr_last_id.data<int64_t>(),
			gr_batch_id.data<int64_t>(),
			gr_h.data<int64_t>(),
			gr_w.data<int64_t>(),
			w,
			batch_size,
			event_size,
			feature_size,
			h*w,
			new_batch_size,
			max_rf_events_cpu);
	}));

	// Decrement all values by one (we want the id, not the count)
	gr_last_id = gr_last_id.sub(1);

    return std::make_tuple(gr_batch_id, gr_last_id, gr_h, gr_w, groups);
}


/*****************************/
/* group_rf_bounded_overlap function */
/*****************************/

template <typename scalar_t>
__global__ void group_rf_bounded_overlap_cuda_kernel(const scalar_t* __restrict__ features,
                                                     const int64_t* __restrict__ pix_idx,
                                                     const int64_t* __restrict__ lengths,
                                                     const int64_t* __restrict__ rf_offsets,
                                                     const int64_t* __restrict__ rf_lengths,
                                                     const int64_t* __restrict__ pix2rf,
                                                     scalar_t* __restrict__ groups,
                                                     int64_t* __restrict__ gr_last_id,
                                                     int64_t* __restrict__ gr_batch_id,
                                                     int64_t* __restrict__ gr_h,
                                                     int64_t* __restrict__ gr_w,
                                                     const int64_t out_w,
                                                     const int64_t batch_size,
                                                     const int64_t event_size,
                                                     const int64_t feature_size,
                                                     const int64_t num_rf,
                                                     const int64_t max_overlap,
                                                     const int64_t new_batch_size){

    const int64_t batch_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_rf_finished = 0;

    if (batch_id < batch_size){
        for (int64_t e = lengths[batch_id] - 1; e >= 0 & num_rf_finished < num_rf; e--){
            // pix_idx[batch_id][e]
            const int64_t pix_id = pix_idx[batch_id * event_size + e];
            // Adds the current event to every RF in which the event is contained
            for (int64_t r = 0; r < max_overlap; r++){
                // pix2rf[pix_id][r]
                const int64_t rf_id = pix2rf[pix_id * max_overlap + r];
                // The list of receptive fields associated to each pixel id is padded with negative
                // values, we stop at the first one.
                if (rf_id < 0) break;
                // rf_lengths[batch_id][rf_id]
                const int64_t rf_len = rf_lengths[batch_id * num_rf + rf_id];
                // Retrieve the position of the receptive field within the output tensor
                // rf_offsets[batch_id][rf_id]
                const int64_t rf_pos = rf_offsets[batch_id * num_rf + rf_id] - 1;

                if (gr_last_id[rf_pos] < rf_len){
                    // Retrieve the number of events already placed inside the target rf
                    // const int64_t event_pos = rf_event_counts[rf_id];
                    const int64_t event_pos = rf_len - 1 - gr_last_id[rf_pos];

                    // groups[event_pos][rf_pos][:] = features[batch_id][e][:]
                    const int64_t write_offset = event_pos * (new_batch_size * feature_size) \
                                                 + rf_pos * feature_size;
                    const int64_t read_offset = batch_id * (event_size * feature_size) \
                                                + e * feature_size;
                    for (int64_t f = 0; f < feature_size; f++)
                        groups[write_offset + f] = features[read_offset + f];

                    // Increment the number of events in the current receptive field
                    gr_last_id[rf_pos] += 1;
                    // Check if the current one was the last event in the current receptive field
                    // in this case increment the number of finished receptive fields
                    if (gr_last_id[rf_pos] == rf_len)
                        num_rf_finished += 1;

                    // We write receptive field information just one time (it is the same for
                    // all the events in the same receptive field), ie, the first time we process
                    // an event inside the receptive field
                    if (event_pos == 0){
                        gr_batch_id[rf_pos] = batch_id;
                        gr_h[rf_pos] = (int)(rf_id / out_w);
                        gr_w[rf_pos] = (int)(rf_id % out_w);
                    }
                }
            }
        }
    }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
group_rf_bounded_overlap_cuda(torch::Tensor features, torch::Tensor pix_idx,
                              torch::Tensor lengths, torch::Tensor pix2rf,
                              torch::Tensor max_rf_events, int h, int w){

    const auto batch_size = features.size(0);
    const auto event_size = features.size(1);
    const auto feature_size = features.size(2);
    const auto max_overlap = pix2rf.size(1);

    // Compute the number of events in each receptive field of every sample
    auto rf_counts = n_rf_events_overlap_cuda(pix_idx, lengths, pix2rf, h*w);
    // Bound the number of events in each receptive field based on the provided max
    if (max_rf_events.cpu().item().to<int64_t>() > 0)
        rf_counts = min2d_scalar_cuda(rf_counts, max_rf_events);

    // Compute the new event length (max events in a receptive field)
    const auto new_event_size = torch::max(rf_counts).cpu().item().to<int64_t>();
    // Compute the position that each non empty receptive field will have in the output tensor
    const auto rf_offsets = torch::cumsum(rf_counts.gt(0).view({-1}), /*dim=*/-1).view(rf_counts.sizes());
    // Compute the new flat batch size (tot num of non empty receptive fields)
    const auto new_batch_size = rf_offsets[batch_size-1][h*w-1].cpu().item().to<int64_t>();

    // Create a tensor to hold output groups
	auto groups = torch::zeros({new_event_size, new_batch_size, feature_size}, features.options());
	auto gr_last_id = torch::zeros({new_batch_size}, features.options().dtype(at::kLong));
	auto gr_batch_id = torch::zeros({new_batch_size}, features.options().dtype(at::kLong));
	auto gr_h = torch::zeros({new_batch_size}, features.options().dtype(at::kLong));
	auto gr_w = torch::zeros({new_batch_size}, features.options().dtype(at::kLong));

	// Allocate a thread for each sample, each one will process all the events
    // in the sample (just the batch loop is parallelized)
	int threadsPerBlock = 32;
	int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

	AT_DISPATCH_ALL_TYPES(features.type(), "group_rf_bounded_overlap_cuda", ([&] {
		group_rf_bounded_overlap_cuda_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
			features.data<scalar_t>(),
			pix_idx.data<int64_t>(),
			lengths.data<int64_t>(),
			rf_offsets.data<int64_t>(),
			rf_counts.data<int64_t>(),
			pix2rf.data<int64_t>(),
			groups.data<scalar_t>(),
			gr_last_id.data<int64_t>(),
			gr_batch_id.data<int64_t>(),
			gr_h.data<int64_t>(),
			gr_w.data<int64_t>(),
			w,
			batch_size,
			event_size,
			feature_size,
			h*w,
			max_overlap,
			new_batch_size);
	}));

	// Decrement all values by one (we want the id, not the count)
	gr_last_id = gr_last_id.sub(1);

    return std::make_tuple(gr_batch_id, gr_last_id, gr_h, gr_w, groups);
}


/**************************/
/* select_last_n function */
/**************************/

template <typename scalar_t>
__global__ void select_n_from_kernel(const scalar_t* __restrict__ input,
                                     scalar_t* __restrict__ output,
                                     const int64_t* __restrict__ batch_from,
                                     const int64_t* __restrict__ batch_n,
                                     const int64_t event_size,
                                     const int64_t batch_size,
                                     const int64_t feature_size){

    const int64_t batch_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_id < batch_size){
        const auto start = batch_from[batch_id];
        for (int64_t e_r = start, e_w = 0; e_w < batch_n[batch_id]; e_r++, e_w++){
            const auto read_offset = (batch_size * feature_size) * e_r + feature_size * batch_id;
            const auto write_offset = (batch_size * feature_size) * e_w + feature_size * batch_id;
            for (int64_t f = 0; f < feature_size; f++)
               output[write_offset + f] = input[read_offset + f];
        }
    }
}

torch::Tensor
select_n_from_cuda(torch::Tensor input, torch::Tensor batch_from, torch::Tensor batch_n){

    const auto event_size = input.size(0);
    const auto batch_size = input.size(1);
	const auto feature_size = input.size(2);

    const auto max_len = torch::max(batch_n).cpu().item().to<int64_t>();
    auto output = torch::zeros({max_len, batch_size, feature_size}, input.options());


    int threadsPerBlock = 32;
	int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

    AT_DISPATCH_ALL_TYPES(input.type(), "select_n_from_cuda", ([&] {
		select_n_from_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
			input.data<scalar_t>(),
			output.data<scalar_t>(),
			batch_from.data<int64_t>(),
			batch_n.data<int64_t>(),
			event_size,
			batch_size,
			feature_size);
	}));

    return output;
}


/******************************/
/* n_interval_events function */
/******************************/

template <typename scalar_t>
__global__ void n_interval_events_cuda_kernel(const scalar_t* __restrict__ ts_percent,
                                              const int64_t* __restrict__ lengths,
                                              int64_t* __restrict__ interval_events_count,
                                              const int64_t batch_size,
                                              const int64_t event_size,
                                              const int64_t n_intervals) {

    const int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
	const float interval_size = 1.0 / n_intervals;

	if (batch_id < batch_size) {
		for (int e = 0; e < event_size & e < lengths[batch_id]; e++) {
			int read_index = batch_id * event_size + e;
			// ts_percent[batch_id][e]
			scalar_t ev_perc = ts_percent[read_index];
			int interval_id = min((long)(ev_perc / interval_size), n_intervals - 1);
			int write_index = batch_id * n_intervals + interval_id;
			// interval_events_count[batch_id][interval_id]
			interval_events_count[write_index] += 1;
		}
	}
}


torch::Tensor n_interval_events_cuda(torch::Tensor ts_percent, torch::Tensor lengths, int n_intervals){

	// rf_idx.shape = [batch_size, n_events]
	const auto batch_size = ts_percent.size(0);
	const auto event_size = ts_percent.size(1);

	// Create a tensor to hold output counts
	auto options = torch::TensorOptions(at::kLong).device(at::kCUDA);
	auto interval_events_count = torch::zeros({batch_size, n_intervals}, options);

	int threadsPerBlock = 32;
	int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

	AT_DISPATCH_ALL_TYPES(ts_percent.type(), "n_interval_events", ([&] {
		n_interval_events_cuda_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
			ts_percent.data<scalar_t>(),
			lengths.data<int64_t>(),
			interval_events_count.data<int64_t>(),
			batch_size,
			event_size,
			n_intervals);
	}));

	return interval_events_count;
}


/*******************************/
/* intervals_to_batch function */
/*******************************/

template <typename scalar_t>
__global__ void intervals_to_batch_cuda_kernel(const scalar_t* events, const int64_t* n_interval_events,
                                               int64_t batch_size, int64_t event_size, int64_t feature_size,
                                               int64_t n_intervals, int64_t new_event_size, scalar_t* new_events) {

    const int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
	const int interval_id = blockIdx.y * blockDim.y + threadIdx.y;
	const int event_id = blockIdx.z * blockDim.z + threadIdx.z;

	if (batch_id < batch_size & interval_id < n_intervals & event_id < new_event_size) {
	    // n_interval_events[batch_id][interval_id]
        int interval_len = n_interval_events[batch_id * n_intervals + interval_id];
	    if (event_id < interval_len){
            int offset = 0;
            for (int i = 0; i < interval_id; i++){
                // n_interval_events[batch_id][i]
                offset += n_interval_events[batch_id * n_intervals + i];
            }
            if ((event_id + offset) < event_size) {
                auto write_offset = batch_id * n_intervals * (new_event_size * feature_size) + \
                                    interval_id * (new_event_size * feature_size) + \
                                    event_id * feature_size;
                auto read_offset = batch_id * (event_size * feature_size) + offset * feature_size + \
                                   event_id * feature_size;

                for (int f = 0; f < feature_size; f++){
                    // new_events[batch_id + interval_id * n_intervals]
                    new_events[write_offset + f] = events[read_offset + f];
                }
            }
        }
	}
}

torch::Tensor intervals_to_batch_cuda(torch::Tensor events, torch::Tensor n_interval_events, torch::Tensor new_event_size){

	// events.shape = [batch_size, n_events, features]
	const auto batch_size = events.size(0);
	const auto event_size = events.size(1);
	const auto feature_size = events.size(2);
	// n_interval_events.shape = [batch_size, n_intervals]
	const auto n_intervals = n_interval_events.size(1);
	const auto new_event_size_cpu = new_event_size.cpu().item().to<int64_t>();

	auto new_events = torch::zeros({batch_size * n_intervals,
                                    new_event_size_cpu, feature_size}, events.options());

	dim3 threadsPerBlock(4, 4, 64);
    dim3 numBlocks((int)((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x),
                   (int)((n_intervals + threadsPerBlock.y - 1) / threadsPerBlock.y),
                   (int)((new_event_size_cpu + threadsPerBlock.z - 1) / threadsPerBlock.z));

	AT_DISPATCH_ALL_TYPES(events.type(), "intervals_to_batch", ([&] {
		intervals_to_batch_cuda_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
			events.data<scalar_t>(),
			n_interval_events.data<int64_t>(),
			batch_size,
			event_size,
			feature_size,
			n_intervals,
			new_event_size_cpu,
			new_events.data<scalar_t>());
	}));

	return new_events;
}