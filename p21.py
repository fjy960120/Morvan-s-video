import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
"""定义添加层"""
def add_layer(inputs,in_size,out_size,n_layer,activitation_function = None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name = 'w')
            tf.summary.histogram(layer_name+'weights',Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
            tf.summary.histogram(layer_name+'biases',biases)
        with tf.name_scope('W_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weights) + biases
        if activitation_function == None:
            outputs = Wx_plus_b
        else:
            outputs = activitation_function(Wx_plus_b)
            tf.summary.histogram(layer_name+'/outputs',outputs)
        return outputs

"""定义输入量数据形式"""
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data =np.square(x_data)-0.5 + noise
#在tensorboard中建立inpus框架
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name = 'x_inpput')
    ys = tf.placeholder(tf.float32,[None,1],name = 'y_input')

"""建立网络"""
#定义隐藏层
l1 = add_layer(xs,1,10,n_layer=1,activitation_function=tf.nn.relu)
#定义输出层
prediction = add_layer(l1,10,1,n_layer=2,activitation_function = None)

"""预测差距"""
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys -prediction),reduction_indices=[1]))
    tf.summary.scalar('loss', loss)

"""开始训练"""
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#运行程序进行训练
init = tf.initialize_all_variables()
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/",sess.graph)
for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 50 == 0:
        result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)
