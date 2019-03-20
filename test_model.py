from p7_TextCNN_model import TextCNN,test
from data_util import get_vocabulary,get_label
import tensorflow as tf

FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("name_scope","cnn","name scope value.")
tf.app.flags.DEFINE_string("training_data_path","../data/ptit-train.txt","path of traning data.") #../data/sample_multiple_label.txt
tf.app.flags.DEFINE_integer("vocab_size",100000,"maximum vocab size.")
tf.app.flags.DEFINE_string("vocab_file_path","../data/Viet22K.txt","path of viet vocab")
vocabulary_word2index, vocabulary_index2word = get_vocabulary(FLAGS.vocab_file_path,name_scope=FLAGS.name_scope)

test(vocabulary_word2index)
