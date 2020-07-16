import tensorflow as tf
from tensorflow.contrib.framework import get_or_create_global_step
from tensorflow.contrib.training import add_gradients_summaries

_GATE_OP = 1
_USE_GLOBAL_STEP = 0


def create_accum_train_op(total_loss,
                          optimizer,
                          optimize_every,
                          global_step=_USE_GLOBAL_STEP,
                          variables_to_train=None,
                          transform_grads_fn=None,
                          summarize_gradients=False,
                          gate_gradients=_GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          check_numerics=True):

    if global_step is _USE_GLOBAL_STEP:
        global_step = get_or_create_global_step()
    tf.summary.scalar("optimizer/global_step", global_step)

    # Creates a variable to keep track of the number of
    # batches seen by the network
    accum_steps = tf.get_variable("accum_steps", initializer=tf.constant(0), trainable=False)
    tf.summary.scalar("optimizer/accum_steps", accum_steps)

    # Defines an operation to update accum_steps
    update_accum_steps = tf.assign_add(accum_steps, 1)

    # Update ops use GraphKeys.UPDATE_OPS collection if update_ops is None.
    global_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if len(global_update_ops) > 0:
        raise RuntimeError("Additional update operations (eg, BatchNorm statistics updates) "
                           "are not supported while accumulating gradients!")

    if variables_to_train is None:
        # Default to tf.trainable_variables()
        variables_to_train = tf.trainable_variables()
    else:
        # Make sure that variables_to_train are in tf.trainable_variables()
        for v in variables_to_train:
            assert v in tf.trainable_variables()

    assert variables_to_train

    with tf.name_scope('train_op'):
        # Make sure total_loss is valid.
        if check_numerics:
            total_loss = tf.check_numerics(total_loss,
                                           'LossTensor is inf or nan')

    # Create the gradients. Note that apply_gradients adds the gradient
    # computation to the current graph and automatically updates global_step
    grads = optimizer.compute_gradients(
        total_loss,
        variables_to_train,
        gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops)

    if transform_grads_fn:
        grads = transform_grads_fn(grads)

    # Create variables to hold accumulated gradients
    # NOTE: we need to initialize variables since some of them (CudnnLSTM)
    #       don't have a static shape yet
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        accum_grads = [tf.Variable(tf.zeros(shape=sess.run(tf.shape(var)), dtype=tf.float32), trainable=False)
                       for var in variables_to_train]

    # Create the operation to accumulate gradients and make sure it is
    # executed after having updated the count of accumulation operations
    with tf.control_dependencies([update_accum_steps, total_loss]):
        accum_ops = [accum_grads[i].assign_add(grad / optimize_every)
                     for i, (grad, var) in enumerate(grads)
                     if grad is not None]
    accum_grads_and_vars = zip(accum_grads, variables_to_train)

    # Summarize gradients.
    if summarize_gradients:
        with tf.name_scope('summarize_grads'):
            add_gradients_summaries(grads)

    def optimize_branch():
        with tf.control_dependencies(accum_ops):
            # Create gradient update operation on accumulated gradients
            apply_grads_op = optimizer.apply_gradients(accum_grads_and_vars, global_step=global_step)
        with tf.control_dependencies([apply_grads_op]):
            # Create an operation to zero the parameter gradients
            # this must always be executed after updating the parameters
            apply_and_zero_grads = [var.assign(tf.zeros_like(var)) for var in accum_grads]
        with tf.control_dependencies(apply_and_zero_grads):
            loss = tf.identity(total_loss)
        return loss

    def accum_branch():
        with tf.control_dependencies(accum_ops):
            loss = tf.identity(total_loss)
        return loss

    # The overall training step
    accum_or_train_op = tf.cond(tf.equal(accum_steps % optimize_every, 0),
                                lambda: optimize_branch(),
                                lambda: accum_branch())

    # Add the operation used for training to the 'train_op' collection
    train_ops = tf.get_collection_ref(tf.GraphKeys.TRAIN_OP)
    if len(train_ops) != 0:
        tf.logging.warning('Emptying the tf.GraphKeys.TRAIN_OP variable')
        del train_ops[:]
        train_ops.append(accum_or_train_op)

    return accum_or_train_op
