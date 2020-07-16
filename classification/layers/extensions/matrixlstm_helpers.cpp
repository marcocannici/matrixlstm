#include <torch/extension.h>

#include <vector>
#include <iostream>


// CUDA declarations

torch::Tensor n_rf_events_cuda(torch::Tensor rf_idx,
                               torch::Tensor lengths,
                               int num_rf);

torch::Tensor n_rf_events_overlap_cuda(torch::Tensor pix_idx,
                                       torch::Tensor lengths,
                                       torch::Tensor pix2rf,
                                       int num_rf);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
group_rf_cuda(torch::Tensor features, torch::Tensor rf_ids,
              torch::Tensor lengths, int h, int w);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
group_rf_bounded_cuda(torch::Tensor features, torch::Tensor rf_ids,
                      torch::Tensor lengths, torch::Tensor max_rf_events,
                      int h, int w, bool keep_most_recent);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
group_rf_bounded_overlap_cuda(torch::Tensor features, torch::Tensor pix_idx,
                              torch::Tensor lengths, torch::Tensor pix2rf,
                              torch::Tensor max_rf_events, int h, int w);

torch::Tensor
select_n_from_cuda(torch::Tensor input, torch::Tensor batch_from, torch::Tensor batch_n);

torch::Tensor
min2d_scalar_cuda(torch::Tensor input, torch::Tensor scalar);

torch::Tensor n_interval_events_cuda(torch::Tensor ts_percent,
                                     torch::Tensor lengths,
                                     int n_intervals);

torch::Tensor intervals_to_batch_cuda(torch::Tensor events,
                                      torch::Tensor n_interval_events,
                                      torch::Tensor new_event_size);

// C++ declarations

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/**
  * Given a tensor of shape [batch_size, n_events] specifying the receptive field associated
  * to every event it computes a tensor of shape [batch_size, num_rf] counting, for each
  * sample in the batch, separately, how many events are associated to each receptive field.
  * The output tensor is the histogram of the receptive field values (rf_idx) computed
  * independently on each sample
  *
  * @param rf_idx: a [batch_size, n_events] tensor containing values in the range [0, num_rf-1]
  * @param  lengths: a [batch_size] tensor specifying the actual (unpadded) length of each
  *                  sample
  * @param num_rf: the number of receptive fields (ie, out_frame.h * out_frame.out_w)
  * @return a [batch_size, num_rf] tensor containing the event receptive fields counts
**/
torch::Tensor n_rf_events(torch::Tensor rf_idx, torch::Tensor lengths, int num_rf){
	CHECK_INPUT(rf_idx);
	CHECK_INPUT(lengths);
	return n_rf_events_cuda(rf_idx, lengths, num_rf);
}


torch::Tensor n_rf_events_overlap(torch::Tensor pix_idx, torch::Tensor lengths,
                                  torch::Tensor pix2rf, int num_rf){
	CHECK_INPUT(pix_idx);
	CHECK_INPUT(lengths);
	CHECK_INPUT(pix2rf);
	return n_rf_events_overlap_cuda(pix_idx, lengths, pix2rf, num_rf);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
group_rf_gpu(torch::Tensor features, torch::Tensor rf_ids,
             torch::Tensor lengths, int h, int w){
    CHECK_INPUT(features);
    CHECK_INPUT(rf_ids);
    CHECK_INPUT(lengths);
    return group_rf_cuda(features, rf_ids, lengths, h, w);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
group_rf_bounded_gpu(torch::Tensor features, torch::Tensor rf_ids,
                     torch::Tensor lengths, torch::Tensor max_rf_events,
                     int h, int w, bool keep_most_recent){
    CHECK_INPUT(features);
    CHECK_INPUT(rf_ids);
    CHECK_INPUT(lengths);
    CHECK_INPUT(max_rf_events);
    return group_rf_bounded_cuda(features, rf_ids, lengths, max_rf_events,
                                 h, w, keep_most_recent);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
group_rf_bounded_overlap_gpu(torch::Tensor features, torch::Tensor pix_idx,
                             torch::Tensor lengths, torch::Tensor pix2rf,
                             torch::Tensor max_rf_events, int h, int w){
    CHECK_INPUT(features);
    CHECK_INPUT(pix_idx);
    CHECK_INPUT(lengths);
    CHECK_INPUT(max_rf_events);
    return group_rf_bounded_overlap_cuda(features, pix_idx, lengths, pix2rf, max_rf_events, h, w);
}

torch::Tensor
select_n_from(torch::Tensor input, torch::Tensor batch_from, torch::Tensor batch_n){
    CHECK_INPUT(input);
    CHECK_INPUT(batch_from);
    CHECK_INPUT(batch_n);
    return select_n_from_cuda(input, batch_from, batch_n);
}

torch::Tensor
min2d_scalar_gpu(torch::Tensor input, torch::Tensor scalar){
    CHECK_INPUT(input);
    CHECK_INPUT(scalar);
    return min2d_scalar_cuda(input, scalar);
}

torch::Tensor
n_interval_events_gpu(torch::Tensor ts_percent, torch::Tensor lengths, int n_intervals){
	CHECK_INPUT(ts_percent);
	CHECK_INPUT(lengths);
	return n_interval_events_cuda(ts_percent, lengths, n_intervals);
}

torch::Tensor
intervals_to_batch_gpu(torch::Tensor events,
                       torch::Tensor n_interval_events,
                       torch::Tensor new_event_size){
    CHECK_INPUT(events);
	CHECK_INPUT(n_interval_events);
	CHECK_INPUT(new_event_size);

	return intervals_to_batch_cuda(events, n_interval_events, new_event_size);
}


/**
 * Given a tensor of events and a receptive field identifier for each of them, group
 * events into different tensors based on their id and sample id. All events in the same
 * sample, having the same id will be placed on the same tensor while maintaining their
 * relative order; events belonging to different samples and having the same receptive
 * field id will also be grouped in different tensors.
 * Given padded events of shape [batch_size, n_pad_events, 4] the returned list of
 * tensors contains at most batch_size * n_receptive_fields tensors, one for each
 * receptive field and sample in the batch. Padding is removed according to the
 * provided lengths while performing the split.
 *
 * @param features: a [batch_size, n_padded_events, 4] tensor containing events features
 * @param rf_ids: a [batch_size, n_padded_events] tensor containing the receptive field
 *                  id associated to every provided event
 * @param  lengths: a [batch_size] tensor specifying the actual (unpadded) length of each
 *                  sample
 * @param h: a int specifying the height of the resulting output frame
 * @param w: a int specifying the width of the resulting output frame
 *
 * @return a tuple containing:
 *          - std::vector<int> gr_batch_id: a list specifying for each output group
 *              which is its original batch sample id
 *          - std::vector<int> gr_last_id: a list specifying the length of each output group
 *          - std::vector<int> gr_h: a list specifying for each tensor in the list which is
 *              its corresponding height position in the output frame
 *          - std::vector<int> gr_w: a list specifying for each tensor in the list which is
 *              its corresponding width position in the output fram
 *          - std::vector<torch::Tensor> groups: a list containing the grouped events
 */
std::tuple<std::vector<int>, std::vector<int>, std::vector<int>,
           std::vector<int>, std::vector<torch::Tensor>>
group_rf(torch::Tensor features, torch::Tensor rf_ids, torch::Tensor lengths, int h, int w) {

    std::vector<torch::Tensor> groups{};
    std::vector<int> gr_batch_id{};
	std::vector<int> gr_last_id{};
    std::vector<int> gr_h{};
    std::vector<int> gr_w{};

    // Retrieves events shape as vector: [batch_size, n_events, n_features]
    std::vector<int64_t> sizes = features.sizes().vec();
    // for each sample in the batch
    for (int64_t b = 0; b < sizes[0]; ++b) {
        int64_t b_length = lengths[b].item().to<int64_t>();

        // Allocate a vector of h*w values containing for the current batch,
        // all the events in each receptive field
        std::vector<std::vector<torch::Tensor>> b_groups(w*h);
        // Instantiate vectors
        for (int64_t v = 0; v < w*h; ++v)
            b_groups.push_back(std::vector<torch::Tensor>());

        // for each event in the sample
        for (int64_t e = 0; (e < sizes[1]) & (e < b_length); ++e){
            int64_t rf_id = rf_ids[b][e].item().to<int64_t>();
            auto feature = features[b][e];
            // Add the event in the corresponding receptive field
            b_groups[rf_id].push_back(feature);
        }

        // b_groups is a list of lists, we convert the inner lists by stacking
        // events within the same receptive field
        // TODO Do we have to free b_groups afterwards?
        std::vector<torch::Tensor> b_tensor_groups{};
        for (int64_t rf = 0; rf < w*h; ++rf){
            if (!b_groups[rf].empty()){
                torch::Tensor stack_tensor = torch::stack(b_groups[rf], 0);

                b_tensor_groups.push_back(stack_tensor);
				gr_batch_id.push_back(b);
				gr_last_id.push_back(b_groups[rf].size()-1);
				gr_h.push_back((int)(rf / w));
				gr_w.push_back((int)(rf % w));
            }
        }

        // Concatenate b_groups to groups vector
        groups.insert( groups.end(), b_tensor_groups.begin(), b_tensor_groups.end() );
    }
    return std::make_tuple(gr_batch_id, gr_last_id, gr_h, gr_w, groups);
}


// Python bindings

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("group_rf", &group_rf, "Group by receptive field");
    m.def("group_rf_gpu", &group_rf_gpu, "Group by receptive field (CUDA)");
    m.def("group_rf_bounded_gpu", &group_rf_bounded_gpu, "Group by receptive field *bounded* (CUDA)");
    m.def("group_rf_bounded_overlap_gpu", &group_rf_bounded_overlap_gpu,
          "Group by overlapping receptive field *bounded* (CUDA)");
    m.def("n_rf_events", &n_rf_events, "Num of receptive field events (CUDA)");
    m.def("n_rf_events_overlap", &n_rf_events_overlap, "Num of receptive field events w/ overlap (CUDA)");
    m.def("select_n_from", &select_n_from, "Select n values from (CUDA)");
    m.def("min2d_scalar_gpu", &min2d_scalar_gpu, "Elem-wise min with scalar (CUDA)");
    m.def("n_interval_events_gpu", &n_interval_events_gpu, "Num of events in each interval (CUDA)");
    m.def("intervals_to_batch_gpu", &intervals_to_batch_gpu, "Intervals to batch (CUDA)");
}