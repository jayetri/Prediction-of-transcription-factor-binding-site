#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import helper
from model import CNN
from sklearn.utils import shuffle
from tensorflow.contrib import learn




# Model Hyperparameters

tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_string("filter_sizes", '3,4,5', "Comma-separated filter sizes (default: '3,4,5')")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 7000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 10, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


#data loading

print("Loading data...")
ftrain=open("proc_train.csv","r")

r=ftrain.read()
r=r.strip().split("\n")
TRY=[]
TRX=[]


for ll in r: 
    TRY.append(ll.split(',')[-2:])
    TRX.append(ll.split(',')[:-2])
ftrain.close()


# Randomly shuffle data

x_shuffled,y_shuffled = shuffle(TRX,TRY)



# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = CNN(
            sequence_length=14,
            num_classes=2,
            embedding_size=4,
            filter_sizes=FLAGS.filter_sizes,
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        oppp = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = oppp.compute_gradients(cnn.loss)
        train_op = oppp.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        summary_gradiented = []
        for a, rs in grads_and_vars:
            if a is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(rs.name), a)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(rs.name), tf.nn.zero_fraction(a))
                summary_gradiented.append(grad_hist_summary)
                summary_gradiented.append(sparsity_summary)
        summary_gradiented_merged = tf.summary.merge(summary_gradiented)

        # Output directory for models and summaries
        ts = str(int(time.time()))
        directory_final = os.path.abspath(os.path.join(os.path.curdir, "runs", ts))
        print("Writing to {}\n".format(directory_final))

        # Summaries for loss and accuracy
        SummaryAccuracy = tf.summary.scalar("accuracy", cnn.accuracy)
        SummaryLoss = tf.summary.scalar("loss", cnn.loss)

        # Train Summaries
        trainingDirectory_Summary = os.path.join(directory_final, "summaries", "train")
        triningOp_Summary = tf.summary.merge([SummaryLoss,  SummaryAccuracy, summary_gradiented_merged])
        train_summary_writer = tf.summary.FileWriter(trainingDirectory_Summary, sess.graph)

        # Dev summaries
        DirSumDev = os.path.join(directory_final, "summaries", "dev")
        OpSummaryDev = tf.summary.merge([SummaryLoss,  SummaryAccuracy])
        SummaryWriterDev = tf.summary.FileWriter(DirSumDev, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        DirectoryCheck = os.path.abspath(os.path.join(directory_final, "checkpoints"))
        PrefixOfCheckpt = os.path.join(DirectoryCheck, "model")
        if not os.path.exists(DirectoryCheck):
            os.makedirs(DirectoryCheck)
        ss = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)


        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_y: y_batch,
              cnn.input_x: x_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, triningOp_Summary, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_y: y_batch,
              cnn.input_x: x_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, OpSummaryDev, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            score=cnn.scores
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy ))
            if writer:
                writer.add_summary(summaries, step)

        bb = helper.batch_iter(
            list(zip(TRX, TRY)), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch...
        for b in bb:
            x_batch, y_batch = zip(*b)
            y_batch=np.resize(np.array(y_batch),(np.shape(x_batch)[0],2))
            x_batch=np.resize(np.array(x_batch),(np.shape(x_batch)[0],14,4,1))
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
           
            if current_step % FLAGS.checkpoint_every == 0:
                path = ss.save(sess, PrefixOfCheckpt, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
