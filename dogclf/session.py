import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

session = tf.compat.v1.Session()
# to ensure the graph is the same across all threads
graph = tf.compat.v1.get_default_graph()