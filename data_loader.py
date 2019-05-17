import numpy as np
import os
import gzip
import cv2
from sklearn.model_selection import train_test_split

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile


DEFAULT_SOURCE_URL_MNIST = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
DEFAULT_SOURCE_URL_FASHION = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'


def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

  Args:
    f: A file object that can be passed into a gzip reader.

  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].

  Raises:
    ValueError: If the bytestream does not start with 2051.

  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 numpy array [index].

  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.

  Returns:
    labels: a 1D uint8 numpy array.

  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels

def load_mnist(data_path, validation_size, source_url, one_hot):
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    local_file = base.maybe_download(TRAIN_IMAGES, data_path,
                                     source_url + TRAIN_IMAGES)
    with gfile.Open(local_file, 'rb') as f:
        train_images = extract_images(f)

    local_file = base.maybe_download(TRAIN_LABELS, data_path,
                                     source_url + TRAIN_LABELS)
    with gfile.Open(local_file, 'rb') as f:
        train_labels = extract_labels(f, one_hot=one_hot)

    local_file = base.maybe_download(TEST_IMAGES, data_path,
                                     source_url + TEST_IMAGES)
    with gfile.Open(local_file, 'rb') as f:
        test_images = extract_images(f)

    local_file = base.maybe_download(TEST_LABELS, data_path,
                                     source_url + TEST_LABELS)
    with gfile.Open(local_file, 'rb') as f:
        test_labels = extract_labels(f, one_hot=one_hot)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'.format(len(train_images), validation_size))

    val_images = train_images[:validation_size]
    val_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    return train_images, train_labels, val_images, val_labels, test_images, test_labels

def load_medical_data(data_path):
    mean_value = np.load(os.path.join(data_path, 'mean_value.npy'))

    train_images = np.load(os.path.join(data_path, 'train_images.npy'))
    train_labels = np.load(os.path.join(data_path, 'train_labels.npy'))

    train_images = normalize_images(train_images, mean_value)

    if 'diaret' in data_path:
        val_images = np.load(os.path.join(data_path, 'test_images.npy'))
        val_labels = np.load(os.path.join(data_path, 'test_labels.npy'))
    else:
        val_images = np.load(os.path.join(data_path, 'val_images.npy'))
        val_labels = np.load(os.path.join(data_path, 'val_labels.npy'))

    val_images = normalize_images(val_images, mean_value)

    test_images = np.load(os.path.join(data_path, 'test_images.npy'))
    test_labels = np.load(os.path.join(data_path, 'test_labels.npy'))

    test_images = normalize_images(test_images, mean_value)

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


def normalize_images(x_images, mean_value):
    """Subtract mean value and normalize images to 0-1."""
    x_flat = np.zeros((x_images.shape[0], 784))
    for k in range(0, x_images.shape[0]):
        img = x_images[k, ...] - mean_value
        img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)
        x_flat[k, ...] = np.reshape(img, [-1])

    return x_flat


class DataSet(object):

  def __init__(self, images, labels, fake_data=False, one_hot=False, reshape=True):
    """Construct a DataSet. one_hot arg is used only if fake_data is true."""

    if fake_data:
      self._num_examples = images.shape[0]
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                  images.shape[1] * images.shape[2])

    # Convert from [0, 255] -> [0.0, 1.0].
    images = images.astype(np.float32)
    images = np.multiply(images, 1.0 / 255.0)

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
      return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """
    Return the next `batch_size` examples from this data set.

    :param batch_size: size of the training mini-batch
    :param fake_data: whether array should be reshaped to image
    """
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in range(batch_size)], [fake_label for _ in range(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1

      # Shuffle data
      np.random.seed(0)
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]

      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples

    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end]


def read_data_sets(data_path, fake_data=False, one_hot=False,
                   validation_size=5000, source_url={},
                   augment=False,
                   percentage_train=100.,
                   unbalance=False,  unbalance_dict={"percentage": 20, "label1": 0, "label2": 8},
                   ):

    """
    Creates a dataset object with information for training, validation and test

    :param data_path: path to directory where data is stored
    :param fake_data: whether array should be reshaped to image
    :param one_hot: whether labels should be one-hot encoded
    :param validation_size: size of validation set
    :param source_url: source containing URL to download mnist data
    :param percentage_train: percentage of training data to be used (experiment: limited data)
    :param unbalance: whether unbalance in the class distribution is desired (experiment: class-imbalance)
    :param unbalance_dict: dictionary containing the parameters that determine the class-distribution
        percentage: amount of data for the reduced selected labels (default -> 20%)
        label1: first selected label to be reduced (default -> digit 0)
        label2: second selected label to be reduced (default -> digit 8)
    """

    class DataSets(object):
        pass

    data_sets = DataSets()

    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True, one_hot=True)
        data_sets.validation = DataSet([], [], fake_data=True, one_hot=True)
        data_sets.test = DataSet([], [], fake_data=True, one_hot=True)
        return data_sets

    if not source_url:  # empty string check
        if 'fashion' in data_path:
            source_url = DEFAULT_SOURCE_URL_FASHION
        else:
            source_url = DEFAULT_SOURCE_URL_MNIST

    if 'fashion' in data_path or 'mnist' in data_path:  # mnist or fashion
        train_images, train_labels, val_images, val_labels, test_images, test_labels = \
            load_mnist(data_path, validation_size, source_url, one_hot)
	reshape = True
    else:
        train_images, train_labels, val_images, val_labels, test_images, test_labels = \
            load_medical_data(data_path)
	reshape = False

    # add random permutation to train & validation
    np.random.seed(42)

    n_train = train_images.shape[0]
    perm = np.random.permutation(n_train)
    train_images = train_images[perm]
    train_labels = train_labels[perm]

    n_val = val_images.shape[0]
    perm = np.random.permutation(n_val)
    val_images = val_images[perm]
    val_labels = val_labels[perm]

    # For experiments with data-augmentation
    if augment:
        if 'fashion' in data_path:  # rotations +-10 and horizontal flips
            augmented_images, augmented_labels = augment_data(train_images, train_labels, hflip=True)
        elif 'mnist' in data_path:  # rotations +-10
            augmented_images, augmented_labels = augment_data(train_images, train_labels, hflip=False)
        train_images = np.concatenate([train_images, np.expand_dims(augmented_images, 3)])
        train_labels = np.concatenate([train_labels, augmented_labels])
        # for the medical datasets, you can use the "augment" argument while doing patch extraction

    # For experiments with limited amount of data
    if percentage_train != 100.:
        train_size = int(0.01*percentage_train*train_images.shape[0])
        Xtrain_images, Xval_images, ytrain, yval = train_test_split(train_images, train_labels, train_size=train_size)
        train_images = Xtrain_images
        train_labels = ytrain

    # For experiments with class-imbalance distribution
    if unbalance:
        n_classes = len(np.unique(np.argmax(train_labels, 1)))
        reduceto = 0.01*unbalance_dict['percentage']
        label1 = unbalance_dict['label1']
        label2 = unbalance_dict['label2']

        pick_ids = []
        newsize = 0
        all_classes = np.arange(0, n_classes)
        all_classes = np.delete(all_classes, np.where(all_classes == label1)[0])
        all_classes = np.delete(all_classes, np.where(all_classes == label2)[0])

        for lab in [label1, label2]:
            allids = np.where(np.argmax(train_labels, 1) == lab)[0]
            selectedids = np.random.choice(allids, int(reduceto * allids.shape[0]), replace=False)
            pick_ids.append(selectedids)
            newsize += len(selectedids)

        new_ids = convert_list_to_array(pick_ids, newsize)

        other_ids = []
        othersize = 0
        for lab in all_classes.tolist():
            selectedids = np.where(np.argmax(train_labels, 1) == lab)[0]
            other_ids.append(selectedids)
            othersize += len(selectedids)

        keep_ids = convert_list_to_array(other_ids, othersize)

        # new_ids: contains the indices of the reduced (imbalance) classes
        # keep_ids: contains the indices of the rest (keep the same class distribution)
        resulting_ids = np.concatenate((new_ids, keep_ids))
        np.random.shuffle(resulting_ids)

        train_images = train_images[resulting_ids, ...]
        train_labels = train_labels[resulting_ids, ...]

    data_sets.train = DataSet(train_images, train_labels, fake_data=True, one_hot=True, reshape=reshape)
    data_sets.validation = DataSet(val_images, val_labels, fake_data=True, one_hot=True, reshape=reshape)
    data_sets.test = DataSet(test_images, test_labels, fake_data=True, one_hot=True, reshape=reshape)

    return data_sets


def augment_data(train_images, train_labels, augmentation_factor=1, hflip=False):
    augmented_image = []
    augmented_image_labels = []

    np.random.seed(0)
    augment_dataset_by = 0.05  # 5%
    all_ids = np.arange(0, train_images.shape[0])
    selected_ids = np.random.choice(all_ids, int(augment_dataset_by*train_images.shape[0]))

    rows = train_images.shape[1]
    cols = train_images.shape[2]

    for num in selected_ids.tolist():
        for i in range(0, augmentation_factor):
            # randomly choose a rotation degree between -10 and 10
            rot_degree = np.random.choice(np.arange(-10, 10, 1), 1)[0]
            # define rotation matrix
            M = cv2.getRotationMatrix2D((rows / 2, rows / 2), rot_degree, 1)  # rotation matrix
            # apply rotation to original image with opencv
            dst = cv2.warpAffine(train_images[num], M, (cols, rows))  # rotated image

            if hflip:  # horizontal flip (left-right)
                if np.random.choice([0, 1]) == 0:
                    dst = np.flip(dst, axis=1)

            augmented_image.append(dst)
            augmented_image_labels.append(train_labels[num])

    return np.array(augmented_image), np.array(augmented_image_labels)


def convert_list_to_array(elements, size):
    array = np.zeros(size, np.int32)
    for kk, ii in enumerate(elements):
        if kk == 0:
            start = 0
            end = len(ii)
        else:
            end += len(ii)
        array[start:end] = ii
        start = end
    return array
