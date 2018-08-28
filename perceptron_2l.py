import tensorflow as tf
import numpy as np
import logging
import copy
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.python import pywrap_tensorflow
import shutil

logger = logging.getLogger('perceptron')

class Perceptron(object):
    def __init__(self, folder, n_gen):
        try:
            self.sess.close()
            tf.reset_default_graph()
        except:
            pass
        #Network data is specified here (no config file), currently prepared for two layers only
        self.n_input = 4
        self.n_hidden_1 = 7
        self.n_hidden_2 = 5
        self.n_output = 1
        #self.saver = tf.train.Saver()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.initialized = False
        self.weights = {'h1': None, 'out': None}
        self.biases = {'b1': None, 'out': None}
        self.fitness = 0
        self.saver = None
        self.folder_model = folder
        self.n_gen = n_gen
        self.x = None
        if self.folder_model == '':
            with tf.variable_scope(str(self.n_gen)):
                self.weights = {
                    'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1]), name='h1'),

                    'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2]), name='h2'),

                    'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_output]), name='out')
                }
                self.biases = {
                    'b1': tf.Variable(tf.random_normal([self.n_hidden_1]), name='b1'),

                    'b2': tf.Variable(tf.random_normal([self.n_hidden_2]), name='b2'),

                    'out': tf.Variable(tf.random_normal([self.n_output]), name='b1.out')
                }
                self.x = tf.placeholder('float', [None, self.n_input],  name='x')

    def multilayer_perceptron(self, X, weights, biases):
        with tf.device("/gpu:0"):
            layer_1 = tf.sigmoid(tf.add(tf.matmul(X, weights['h1']), biases['b1']))
            layer_2 = tf.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))

            return tf.sigmoid(tf.matmul(layer_2, weights['out']) + biases['out'])

    #Initialize network

    def init1(self):
        self.sess.close()
        tf.reset_default_graph()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        if self.folder_model == '': #Create new
            with tf.variable_scope(str(self.n_gen)):
                self.weights = {
                    'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1]), name='h1'),

                    'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2]), name='h2'),

                    'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_output]), name='out')
                }
                self.biases = {
                    'b1': tf.Variable(tf.random_normal([self.n_hidden_1]), name='b1'),

                    'b2': tf.Variable(tf.random_normal([self.n_hidden_2]), name='b2'),

                    'out': tf.Variable(tf.random_normal([self.n_output]), name='b1.out')
                }
                self.x = tf.placeholder('float', [None, self.n_input],  name='x')
                self.pred = self.multilayer_perceptron(self.x, self.weights, self.biases)
                # if not self.sess:
                #self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
                self.init = tf.global_variables_initializer()
                self.sess.run(self.init)
                self.saver = tf.train.Saver()

        else: #Get from folder.

            self.sess.run(tf.global_variables_initializer())

            path = self.folder_model + '/' + str(self.n_gen) + '/model'

            previous_scope = self.get_previous_scope(path)

            self.new_saver = None
            self.new_saver = tf.train.import_meta_graph(path+'.meta')
            #try:
            self.new_saver.restore(self.sess, tf.train.latest_checkpoint(self.folder_model+'/'+str(self.n_gen)))
            print_tensors_in_checkpoint_file(tf.train.latest_checkpoint(self.folder_model+'/'+str(self.n_gen)), all_tensors= True, tensor_name='')
            logger.info(f'Restored model {self.n_gen}')

            #self.sess.run(tf.local_variables_initializer())
            with tf.variable_scope(str(self.n_gen)):
                graph = tf.get_default_graph()
                self.weights['h1'] = graph.get_tensor_by_name(previous_scope+"/h1:0")
                self.weights['h2'] = graph.get_tensor_by_name(previous_scope + "/h2:0")
                self.weights['out'] = graph.get_tensor_by_name(previous_scope+"/out:0")
                self.biases['b1'] = graph.get_tensor_by_name(previous_scope+"/b1:0")
                self.biases['b2'] = graph.get_tensor_by_name(previous_scope + "/b2:0")
                self.biases['out'] = graph.get_tensor_by_name(previous_scope+"/b1.out:0")

                self.x = tf.placeholder('float', [None, self.n_input], name="x")
            self.pred = self.multilayer_perceptron(self.x, self.weights, self.biases)

        self.get_dict()
        self.initialized = True

    def get_previous_scope(self,  path):
        reader = pywrap_tensorflow.NewCheckpointReader(path)
        variables = reader.get_variable_to_shape_map()

        for key in variables:
            previous_scope = key.split('/')[0]

        return previous_scope

    #Activate the network with inputs
    def activate(self, inputs):
        if self.initialized is False:
            self.init1()
            self.initialized = True
        with tf.device("/gpu:0"):
            outputs = self.sess.run(self.pred, feed_dict={self.x: inputs})

        return outputs

    def get_dict(self): #Outputs a dict with the weights and biases of the network
        #self.sess = tf.Session()
        with tf.device("/gpu:0"):
            arr3 = tf.reshape(self.weights['out'],[self.n_hidden_2*self.n_output]).eval(session=self.sess)
            arr2 = tf.reshape(self.weights['h2'], [self.n_hidden_1 * self.n_hidden_2]).eval(session=self.sess)
            arr1 = tf.reshape(self.weights['h1'], [self.n_input * self.n_hidden_1]).eval(session=self.sess)
        weight_arr = np.hstack((arr1, arr2, arr3))
        #weight_arr = np.append(weight_arr, arr3)
        biases_arr = np.hstack((self.biases['b1'].eval(session=self.sess),self.biases['b2'].eval(session=self.sess), self.biases['out'].eval(session=self.sess)))
        self.weights_arr = weight_arr
        self.biases_arr = biases_arr
        self.as_dict = {"weights":weight_arr,"biases":biases_arr}

        return self.as_dict

    def reload(self): #Reload a network with a previous dict (after crossover and mutation)
        self.sess.close()
        tf.reset_default_graph()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())
        weights_arr = self.as_dict['weights']
        biases_arr = self.as_dict['biases']
        dim1 = self.n_input * self.n_hidden_1
        dim2 = self.n_hidden_2 * self.n_hidden_1 + dim1

        dim1_bias = self.n_hidden_1
        dim2_bias = self.n_hidden_1+self.n_hidden_2

        with tf.variable_scope(str(self.n_gen)):
            self.weights['h1'] = tf.Variable(np.reshape(weights_arr[:dim1], (self.n_input, self.n_hidden_1)), name='h1')
            self.weights['h1'].load(np.reshape(weights_arr[:dim1], (self.n_input, self.n_hidden_1)), self.sess)

            self.weights['h2'] = tf.Variable(np.reshape(weights_arr[dim1:dim2], (self.n_hidden_1, self.n_hidden_2)), name='h2')
            self.weights['h2'].load(np.reshape(weights_arr[dim1:dim2], (self.n_hidden_1, self.n_hidden_2)), self.sess)

            self.weights['out'] = tf.Variable(np.reshape(weights_arr[dim2:], (self.n_hidden_2, self.n_output)), name='out')
            self.weights['out'].load(np.reshape(weights_arr[dim2:], (self.n_hidden_2, self.n_output)), self.sess)

            self.biases['b1'] = tf.Variable(biases_arr[:dim1_bias], name='b1')
            self.biases['b1'].load(biases_arr[:dim1_bias], self.sess)

            self.biases['b2'] = tf.Variable(biases_arr[dim1_bias:dim2_bias], name='b2')
            self.biases['b2'].load(biases_arr[dim1_bias:dim2_bias], self.sess)

            self.biases['out'] = tf.Variable(biases_arr[dim2_bias:], name='b1.out')
            self.biases['out'].load(biases_arr[dim2_bias:], self.sess)

            self.x = tf.placeholder('float', [None, self.n_input], name='x')

        self.pred = self.multilayer_perceptron(self.x, self.weights, self.biases)

        self.initialized = True

    def copy(self): #Copy newtorks
        d = copy.deepcopy(self.as_dict)
        p = Perceptron(self.folder_model, self.n_gen)
        p.as_dict = d
        return p

    def save_net(self):
        try:
            shutil.rmtree('./tmp/'+str(self.n_gen), ignore_errors=True)
        except:
            pass

        with tf.variable_scope(str(self.n_gen)):
            self.saver = tf.train.Saver([self.weights['h1'], self.weights['h2'], self.weights['out'], self.biases['b1'], self.biases['b2'], self.biases['out']])
            #self.sess.run(self.init)

            path = './tmp/'+str(self.n_gen)+'/model'
            save_path = self.saver.save(self.sess, path)
            logger.info(f'Modelo guardado en {save_path}')