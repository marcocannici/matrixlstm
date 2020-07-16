#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"


/********************/
/* helper functions */
/********************/

int ReadGPUIntegerScalar(const int* tensor_data) {

	int host_memory;
    cudaMemcpy(&host_memory,
               tensor_data,
               sizeof(int),
               cudaMemcpyDeviceToHost);

    return host_memory;
}

__global__ void Zero1DKernel(int* input, int N) {

	const int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n < N)
	    input[n] = 0;
}

void Zero1DKernelLauncher(int* input, int N) {

	int threadsPerBlock = 1024;
	int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

	Zero1DKernel<<<numBlocks, threadsPerBlock>>>(input, N);
}

__global__ void Zero2DKernel(int* input, int N, int M) {

	const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (n < N & m < M)
	    input[n*M + m] = 0;
}

void Zero2DKernelLauncher(int* input, int N, int M) {

	dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((int)((N + threadsPerBlock.x - 1) / threadsPerBlock.x),
                   (int)((M + threadsPerBlock.y - 1) / threadsPerBlock.y));

	Zero2DKernel<<<numBlocks, threadsPerBlock>>>(input, N, M);
}

__global__ void Zero3DKernel(float* input, int N, int M, int K) {

	const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (n < N & m < M & k < K)
	    input[n*M*K + m*K + k] = 0;
}

void Zero3DKernelLauncher(float* input, int N, int M, int K) {

	dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((int)((N + threadsPerBlock.x - 1) / threadsPerBlock.x),
                   (int)((M + threadsPerBlock.y - 1) / threadsPerBlock.y),
                   (int)((K + threadsPerBlock.z - 1) / threadsPerBlock.z));

	Zero3DKernel<<<numBlocks, threadsPerBlock>>>(input, N, M, K);
}

/************************/
/* n_rf_events function */
/************************/

__global__ void NRfEventsKernel(const int* rf_idx,
								const int* lengths,
                                int batch_size,
								int event_size,
								int num_rf,
                                int* rf_events_count) {

	const int batch_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (batch_id < batch_size) {
		for (int e = 0; e < event_size & e < lengths[batch_id]; e++) {
			int read_index = batch_id * event_size + e;
			// rf_idx[batch_id][e]
			int rf_id = rf_idx[read_index];
			if (rf_id == -1)
                continue;

			int write_index = batch_id * num_rf + rf_id;
			// rf_events_count[batch_id][rf_id]
			rf_events_count[write_index] += 1;
		}
	}
}

void NRfEventsKernelLauncher(const int* rf_idx,
							 const int* lengths,
						 	 int batch_size, 
						 	 int event_size, 
						 	 int num_rf, 
							 int* rf_events_count) {

	int threadsPerBlock = 32;
	int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
 
	NRfEventsKernel<<<numBlocks, threadsPerBlock>>>(
		rf_idx, lengths, batch_size, 
		event_size, num_rf, 
		rf_events_count);
}


/*****************************/
/* group_rf_bounded function */
/*****************************/

__global__ void GroupRfBoundedKernel(const float* features, const int* rf_ids, const int* lengths,
                                     const int* rf_offsets, const int* rf_lengths, float* groups,
                                     int* gr_last_id, int* gr_batch_id, int* gr_h, int* gr_w,
                                     const int out_w, const int out_h, const int batch_size,
                                     const int event_size, const int feature_size,
                                     const int new_batch_size, const int max_rf_events) {

	const int64_t batch_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_rf_finished = 0;
    int num_rf = out_h * out_w;

    if (batch_id < batch_size && lengths[batch_id] > 0){
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
                if (gr_last_id[rf_pos] == rf_len){
                    // If the current one was the last event in the current receptive field
                    //  increment the number of finished receptive fields
                    num_rf_finished += 1;

                    // We write receptive field information just one time (it is the same for
                    // all the events in the same receptive field), ie, the last time we process
                    // an event inside the receptive field
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

void GroupRfBoundedKernelLauncher(const float* features, const int* rf_ids, const int* lengths,
                                  const int* rf_offsets, const int* rf_counts, float* groups,
                                  int* gr_last_id, int* gr_batch_id, int* gr_h, int* gr_w,
                                  const int out_w, const int out_h, const int batch_size,
                                  const int event_size, const int feature_size,
                                  const int new_batch_size, const int max_rf_events){

	int threadsPerBlock = 32;
	int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

	GroupRfBoundedKernel<<<numBlocks, threadsPerBlock>>>(
		features, rf_ids, lengths, rf_offsets, rf_counts,
		groups, gr_last_id, gr_batch_id, gr_h, gr_w,
        out_w, out_h, batch_size, event_size, feature_size,
        new_batch_size, max_rf_events);
}


/***************************/
/* sparse_adj_idx function */
/***************************/

__global__ void SparseAdjIdxKernel(const int* rf_ev_step, const int* rf_offsets,
                                   const int* rf_lens, const int num_rf,
                                   const int max_rf_len, int* sparse_idx) {

    const int rf_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int centroid_id = blockIdx.y * blockDim.y + threadIdx.y;

    if (rf_id < num_rf & centroid_id < max_rf_len){
        auto rf_step = rf_ev_step[rf_id];
        auto rf_len = rf_lens[rf_id];
        auto rf_offset = rf_offsets[rf_id];
        int half_ev_step = rf_step / 2;
        int64_t centroid_pos = half_ev_step + centroid_id * rf_step;

        auto right_bound = half_ev_step;
        // If the step is even the right side has one element less
        if (rf_step % 2 == 0)
            right_bound = half_ev_step - 1;
        // Always extend the last neighborhood to the end of the sequence
        if (centroid_id == max_rf_len - 1)
            right_bound = rf_len - centroid_pos - 1;
        for (int neigh_offset = -half_ev_step; neigh_offset <= right_bound; neigh_offset++){
            int64_t write_pos = (rf_offset + centroid_pos + neigh_offset) * 3;
            sparse_idx[write_pos] = rf_id;
            sparse_idx[write_pos + 1] = centroid_id;
            sparse_idx[write_pos + 2] = centroid_pos + neigh_offset;
        }
    }
}

void SparseAdjIdxKernelLauncher(const int* rf_ev_step, const int* rf_offsets,
                                const int* rf_lens, const int num_rf,
                                const int max_rf_len, int* sparse_idx){

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((int)((num_rf + threadsPerBlock.x - 1) / threadsPerBlock.x),
                   (int)((max_rf_len + threadsPerBlock.y - 1) / threadsPerBlock.y));

	SparseAdjIdxKernel<<<numBlocks, threadsPerBlock>>>(rf_ev_step, rf_offsets,
	                                                   rf_lens, num_rf,
	                                                   max_rf_len, sparse_idx);
}


/**************************/
/* get_centroids function */
/**************************/

__global__ void GetCentroidsKernel(const float* events, const int* rf_ev_step,
                                   const int max_rf_len, const int num_rf,
                                   const int event_size, const int feature_size,
                                   float* centroids) {

    const int rf_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int centroid_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel_id = blockIdx.z * blockDim.z + threadIdx.z;

    if (rf_id < num_rf & centroid_id < max_rf_len & channel_id < feature_size){
        auto rf_step = rf_ev_step[rf_id];
        int half_ev_step = rf_step / 2;
        int64_t centroid_pos = half_ev_step + centroid_id * rf_step;
        // events[rf_id][centroid_pos][channel_id]
        int64_t read_pos = rf_id * (event_size * feature_size) \
                            + centroid_pos * (feature_size) \
                            + channel_id;
        // centroids[rf_id][centroid_id][channel_id]
        int64_t write_pos = rf_id * (max_rf_len * feature_size) \
                             + centroid_id * (feature_size) \
                             + channel_id;

        centroids[write_pos] = events[read_pos];
    }
}

void GetCentroidsKernelLauncher(const float* events, const int* rf_ev_step,
                                const int max_rf_len, const int num_rf,
                                const int event_size, const int feature_size,
                                float* centroids){

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((int)((num_rf + threadsPerBlock.x - 1) / threadsPerBlock.x),
                   (int)((max_rf_len + threadsPerBlock.y - 1) / threadsPerBlock.y),
                   (int)((feature_size + threadsPerBlock.z - 1) / threadsPerBlock.z));

	GetCentroidsKernel<<<numBlocks, threadsPerBlock>>>(events, rf_ev_step,
	                                                   max_rf_len, num_rf,
	                                                   event_size, feature_size,
	                                                   centroids);
}


/******************************/
/* n_interval_events function */
/******************************/

__global__ void NIntervalEventsKernel(const float* ts_percent, const int* lengths,
                                      int batch_size, int event_size, int n_intervals,
                                      int* interval_events_count) {

	const int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
	const float interval_size = 1.0 / n_intervals;

	if (batch_id < batch_size) {
		for (int e = 0; e < event_size & e < lengths[batch_id]; e++) {
			int read_index = batch_id * event_size + e;
			// ts_percent[batch_id][e]
			float ev_perc = ts_percent[read_index];
			int interval_id = min((int)(ev_perc / interval_size), n_intervals - 1);
			int write_index = batch_id * n_intervals + interval_id;
			// interval_events_count[batch_id][interval_id]
			interval_events_count[write_index] += 1;
		}
	}
}

void NIntervalEventsKernelLauncher(const float* ts_percent, const int* lengths,
                                   int batch_size, int event_size, int n_intervals,
                                   int* interval_events_count){

	int threadsPerBlock = 32;
	int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

	NIntervalEventsKernel<<<numBlocks, threadsPerBlock>>>(
		ts_percent, lengths, batch_size,
		event_size, n_intervals,
		interval_events_count);
}


/*******************************/
/* intervals_to_batch function */
/*******************************/

__global__ void IntervalsToBatchKernel(const float* events, const int* n_interval_events,
                                       int batch_size, int event_size, int feature_size,
                                       int n_intervals, int new_event_size, float* new_events) {

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

void IntervalsToBatchKernelLauncher(const float* events, const int* n_interval_events,
                                    int batch_size, int event_size, int feature_size,
                                    int n_intervals, int new_event_size, float* new_events){

	dim3 threadsPerBlock(4, 4, 64);
    dim3 numBlocks((int)((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x),
                   (int)((n_intervals + threadsPerBlock.y - 1) / threadsPerBlock.y),
                   (int)((new_event_size + threadsPerBlock.z - 1) / threadsPerBlock.z));

	IntervalsToBatchKernel<<<numBlocks, threadsPerBlock>>>(
		events, n_interval_events,
		batch_size, event_size,
		feature_size, n_intervals,
		new_event_size, new_events);
}

#endif
