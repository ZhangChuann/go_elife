# coding:utf-8
#copy from https://github.com/hzy46/Deep-Learning-21-Examples/blob/master/chapter_1/download.py
import dataset
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.platform import gfile
import os
from six.moves.urllib.request import urlretrieve
from tensorflow.python.util.deprecation import deprecated

WORK_DIRECTOTY = "../dataset/mnist"
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
#WORK_DIRECTOTY = "/home/zc/workspace/go_elife/AI_Demo/dataset/mnist"
train = dataset.train(WORK_DIRECTOTY)
test = dataset.test(WORK_DIRECTOTY)

def maybe_download_1(filename, work_directory, source_url):
  """Download the data from source url, unless it's already here.

  Args:
      filename: string, name of the file in the directory.
      work_directory: string, path to working directory.
      source_url: url to download from if file doesn't exist.

  Returns:
      Path to resulting file.
  """
  if not gfile.Exists(work_directory):
    gfile.MakeDirs(work_directory)
  filepath = os.path.join(work_directory, filename)
  print(filepath)
  if not gfile.Exists(filepath):
    temp_file_name, _ = urlretrieve(source_url)
    gfile.Copy(temp_file_name, filepath)
    with gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath


def maybe_download(filename, work_directory, source_url):
    """A helper to download the data files if not present."""
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urlretrieve(source_url + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    else:
        print('Already downloaded', filename)
        print(filepath)
    return filepath

train_data_filename = maybe_download('train-images-idx3-ubyte.gz', WORK_DIRECTOTY, SOURCE_URL)
train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz', WORK_DIRECTOTY, SOURCE_URL)
test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz', WORK_DIRECTOTY, SOURCE_URL)
test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz', WORK_DIRECTOTY, SOURCE_URL)


mnist = input_data.read_data_sets(WORK_DIRECTOTY, one_hot=True)
print(mnist.train.images.shape)  #(55000,784)
print(mnist.train.labels.shape)    #(55000,10)


print(mnist.validation.images.shape)  #(5000,784)
print(mnist.validation.labels.shape)    #(5000,10)

print(mnist.test.images.shape)  #(10000,784)
print(mnist.test.labels.shape)    #(10000,10)

print(mnist.train.images[0, :])
print(mnist.train.labels[0, :])

