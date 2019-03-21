from p7_TextCNN_model import TextCNN,test
from data_util import get_vocabulary,get_label,load_data_multilabel,load_test_data
import tensorflow as tf

FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("name_scope","cnn","name scope value.")
tf.app.flags.DEFINE_string("test_data_path","../data/ptit-test.txt","path of test data.")
tf.app.flags.DEFINE_string("training_data_path","../data/ptit-train.txt","path of traning data.") #../data/sample_multiple_label.txt
tf.app.flags.DEFINE_integer("vocab_size",100000,"maximum vocab size.")
tf.app.flags.DEFINE_string("vocab_file_path","../data/Viet22K.txt","path of viet vocab")
vocabulary_word2index, vocabulary_index2word = get_vocabulary(FLAGS.vocab_file_path,name_scope=FLAGS.name_scope)
vocabulary_label2index,vocabulary_index2label = get_label(FLAGS.training_data_path,name_scope=FLAGS.name_scope)
#print(vocabulary_word2index)
test_data = load_test_data(FLAGS.test_data_path,vocabulary_word2index,vocabulary_label2index,sentence_len=5,training_portion=0.0)
print("testdata",test_data)
test(test_data)
