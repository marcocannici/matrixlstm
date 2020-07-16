import os
import tensorflow as tf

_cuda_op_module = tf.load_op_library(
    os.path.join(os.path.dirname(__file__), 'matrixlstm_helpers.so'))
_group_rf_bounded = _cuda_op_module.group_rf_bounded
_n_rf_events = _cuda_op_module.n_rf_events
_sparse_adj_idx = _cuda_op_module.sparse_adj_idx
_get_centroids = _cuda_op_module.get_centroids
_n_interval_events = _cuda_op_module.n_interval_events
_intervals_to_batch = _cuda_op_module.intervals_to_batch


# Source: https://github.com/tensorflow/tensorflow/issues/9210#issuecomment-497889961
def sparse_dense_matmult_batch(sp_a, b, back_prop=False):

    def map_function(x):
        i, dense_slice = x[0], x[1]
        sparse_slice = tf.sparse_reshape(tf.sparse_slice(sp_a, [i, 0, 0],
                                                         [1, sp_a.dense_shape[1], sp_a.dense_shape[2]]),
                                         [sp_a.dense_shape[1], sp_a.dense_shape[2]])
        mult_slice = tf.sparse_tensor_dense_matmul(sparse_slice, dense_slice)
        return mult_slice

    elems = (tf.range(0, sp_a.dense_shape[0], delta=1, dtype=tf.int64), b)
    return tf.map_fn(map_function, elems, dtype=tf.float32,
                     parallel_iterations=2048, back_prop=back_prop)


def group_rf_bounded(input, rf_idx, lengths, max_events_per_rf, h, w,
                     keep_most_recent, add_step):

    assert max_events_per_rf > 0, "max_events_per_rf should be > 0"

    with tf.name_scope("GroupRfBoundedWrapper"):
        rf_counts = _n_rf_events(rf_idx, lengths, num_rf=w*h)
        bounded_rf_counts = rf_counts
        if max_events_per_rf and max_events_per_rf > 0:
            bounded_rf_counts = tf.minimum(rf_counts, max_events_per_rf)
        new_event_size = tf.expand_dims(tf.reduce_max(bounded_rf_counts), 0)
        # cumsum is performed using floats for GPU compatibility
        rf_offsets = tf.cumsum(tf.reshape(tf.cast(bounded_rf_counts > 0, tf.float32), [-1]))
        rf_offsets = tf.cast(rf_offsets, tf.int32)
        new_batch_size = tf.expand_dims(rf_offsets[-1], 0)
        rf_offsets = tf.reshape(rf_offsets, bounded_rf_counts.shape)
        max_events_per_rf = tf.constant([max_events_per_rf])

        if keep_most_recent:
            rf_counts = bounded_rf_counts

        group_res = _group_rf_bounded(input, rf_idx, lengths,
                                      rf_offsets, rf_counts,
                                      new_event_size,
                                      new_batch_size,
                                      h=h, w=w,
                                      max_rf_events=max_events_per_rf)
        gr_batch_id, gr_last_id, gr_h, gr_w, groups = group_res
        gr_last_id -= 1

        if not keep_most_recent and add_step:
            # Compute the padding that has been applied internally by
            # _group_rf_bounded.
            # rf_counts.shape = [batch_size, num_rf]
            valid_rf_mask = rf_counts > 0
            valid_rf_len = tf.boolean_mask(rf_counts, valid_rf_mask)
            ev_steps = valid_rf_len // (max_events_per_rf - 1)
            # [1, new_batch_size, 1]
            ev_steps = tf.reshape(ev_steps, [1, -1, 1])
            # [new_event_size, new_batch_size, 1]
            tile_shape = tf.concat([new_event_size, tf.constant([1, 1])], 0)
            ev_steps = tf.cast(tf.tile(ev_steps, tile_shape), groups.dtype)
            groups = tf.concat([groups, ev_steps], -1)

    return gr_batch_id, gr_last_id, gr_h, gr_w, groups


def intervals_to_batch(events, lengths, n_intervals, ts_idx=2):
    # Computes the ts percentage of each event
    ts_mins = tf.reshape(events[:, 0, ts_idx], [-1, 1])
    ts_maxs = tf.reshape(tf.reduce_max(events[:, :, ts_idx], axis=-1), [-1, 1])
    ts_percentage = (events[:, :, ts_idx] - ts_mins) / (ts_maxs - ts_mins + 1e-5)
    ts_percentage = tf.debugging.check_numerics(ts_percentage, "NaN or Inf in ts_percentage")
    # shape: [batch_size, n_intervals]
    inter_ev_count = _n_interval_events(ts_percentage, lengths, n_intervals)
    new_lengths = tf.reshape(inter_ev_count, [-1])
    new_event_size = tf.reduce_max(new_lengths)
    new_events = _intervals_to_batch(events, inter_ev_count,
                                     tf.expand_dims(new_event_size, 0))
    return new_events, new_lengths


if __name__ == "__main__":

    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))

    import pudb; pudb.set_trace()
    events = tf.reshape(tf.range(3 * 10 * 4, dtype=tf.float32), [3, 10, 4])
    lengths = tf.constant([10, 10, 10])
    n_intervals = 4
    new_events, new_lengths = intervals_to_batch(events, lengths, n_intervals)
    print(new_events.eval())
    quit()

    rf_lens = tf.constant([9, 1, 17])
    rf_ev_step = tf.constant([3, 0, 5])
    rf_offsets = tf.constant([0, 9, 10])
    out_dim = tf.expand_dims(tf.reduce_sum(rf_lens), 0)
    max_rf_len = 3
    idx = _sparse_adj_idx(rf_ev_step, rf_offsets, rf_lens, out_dim, max_rf_len)
    sparse_tensor = tf.SparseTensor(indices=tf.cast(idx, tf.int64),
                                    values=tf.ones(out_dim),
                                    dense_shape=[3, 3, 17])  # [B, MAXrf, Tpad]
    dense_tensor = tf.sparse_to_dense(tf.cast(idx, tf.int64),
                                      [3, 3, 17],
                                      tf.ones(out_dim))
    events = tf.reshape(tf.range(3*17*2), [3,17,2])

    a_sp = tf.SparseTensor(indices=[[0,0,0], [0,1,1], [1,0,1], [1,1,0]],
                           values=[1.,1.,1.,1.],
                           dense_shape=[2,2,2])
    b = tf.cast(tf.reshape(tf.range(2*2), [2,2,1]), tf.float32)

    dot = sparse_dense_matmult_batch(a_sp, b)

    max_events_per_rf = 2
    events = tf.reshape(tf.range(3*5*2, dtype=tf.float32), [3, 5, 2])
    rf_lens = tf.constant([2, 1, 5])
    rf_ev_steps = tf.maximum(rf_lens // max_events_per_rf, 1)
    centroids_fts = _get_centroids(events, rf_ev_steps, max_events_per_rf)

    shape = 5
    events = tf.constant(tf.random_uniform([2, 10, 4], 0, shape, dtype=tf.int32).eval(), dtype=tf.float32)
    lengths = tf.constant([10, 10])
    rf_idx = tf.cast(events[..., 0] * shape + events[..., 1], tf.int32)
    num_rf = shape * shape

    import pudb; pudb.set_trace()
    counts = _n_rf_events(rf_idx, lengths, num_rf)
    print(counts.eval())
    with tf.device('/device:GPU:0'):
        group_res = group_rf_bounded(events, rf_idx, lengths,
                                     max_events_per_rf,
                                     shape, shape,
                                     True, False)


    print(group_res.eval())
    print(idx.shape)
    print(idx.eval())
    print(dot.eval())
