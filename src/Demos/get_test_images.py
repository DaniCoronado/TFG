import tensorflow as tf
import random
import sys
sys.path.insert(1, 'AQA-framework-dev_coteach')

from datasets import AVA_generators
random.seed(42)

test = AVA_generators()

dataset = tf.data.Dataset.from_tensor_slices((test.train_image_paths, test.train_scores))

low_quality = []
mid_quality = []
high_quality = []

for filename, label in dataset:
 label = tf.get_static_value(label)
 filename = tf.get_static_value(filename)
 if label[0] < 0.3:
  low_quality.append((filename, label[0]))
 elif label[0] < 0.6:
  mid_quality.append((filename, label[0]))
 else: high_quality.append((filename, label[0]))
 
print(" Low Quality Test\n", random.sample(set(low_quality), 2), "\n\n", "Mid Quality Test\n", random.sample(set(mid_quality), 2), "\n\n", "High Quality Test\n", random.sample(set(high_quality), 2))
