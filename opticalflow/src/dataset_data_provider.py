"""Has the same behaviour of tf.contrib.slim.dataset_data_provider.DatasetDataProvider
but it also exposes the reader_kwargs argument of parallel_read(), allowing to create
the dataset class with specific options (eg, GZip encoding)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.slim.python.slim.data import data_provider
from tensorflow.contrib.slim.python.slim.data import parallel_reader


class DatasetDataProvider(data_provider.DataProvider):

  def __init__(self, dataset, num_readers=1, shuffle=True, num_epochs=None,
               common_queue_capacity=256, common_queue_min=128, reader_kwargs=None):
    """Creates a DatasetDataProvider.
    Args:
      dataset: An instance of the Dataset class.
      num_readers: The number of parallel readers to use.
      shuffle: Whether to shuffle the data sources and common queue when
        reading.
      num_epochs: The number of times each data source is read. If left as None,
        the data will be cycled through indefinitely.
      common_queue_capacity: The capacity of the common queue.
      common_queue_min: The minimum number of elements in the common queue after
        a dequeue.
    """
    _, data = parallel_reader.parallel_read(
        dataset.data_sources,
        reader_class=dataset.reader,
        num_epochs=num_epochs,
        num_readers=num_readers,
        reader_kwargs=reader_kwargs,
        shuffle=shuffle,
        capacity=common_queue_capacity,
        min_after_dequeue=common_queue_min)

    items = dataset.decoder.list_items()
    tensors = dataset.decoder.decode(data, items)

    super(DatasetDataProvider, self).__init__(
        items_to_tensors=dict(zip(items, tensors)),
        num_samples=dataset.num_samples)