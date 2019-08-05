import tensorflow as tf
matrix1 = ([[3,3]])
matrix2 = ([[2],[2]])
prodect = tf.matmul(matrix1,matrix2)
"""Session用法1"""
#sess = tf.Session()
#result = sess.run(prodect)
#print(result)
"""Session用法2"""
with tf.Session() as sess:
    result2 = sess.run(prodect)
    print(result2)
