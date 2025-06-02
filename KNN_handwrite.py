X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
Y = tf.placeholder(dtype=tf.float32, shape=[784])

distance = tf.reduce_sum(tf.sqrt(tf.pow(X-Y, 2)), axis=1)
sorted_distance = tf.contrib.framework.sort(distance)
top_k = tf.slice(sorted_distance, begin=[0], size=[7])


init_op = [tf.global_variables_initializer()]
prediction = []
with tf.Session() as sess:
    sess.run(init_op)
    
    for i in range(len(test_feature)):
        dist, min_k_dist = sess.run([distance, top_k], feed_dict={
            X: train_feature,
            Y: test_feature[i]
        })
        idx = [dist.tolist().index(i) for i in min_k_dist]
        counter = Counter(train_label[idx])
        prediction.append({
            'prediction': counter.most_common(1)[0][0], 
            'label': test_label[i]})
