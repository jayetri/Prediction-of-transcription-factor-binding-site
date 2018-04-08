import tensorflow as tf
import numpy as np
import os
import time
import datetime
from model import CNN
import csv
import helper

# Parameters
# ==================================================


# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "/home/jayetri/Desktop/gene/runs/1523224519/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
print("\nParameters:")
for a, v in sorted(FLAGS.__flags.items()):
    print("{}={}".format(a.upper(), v))
print("")


ftest=open("proc_test.csv","r")

A=ftest.read()
A=A.strip().split("\n")
x_test=[]
for aa in A:
    aa=aa.split(",")
    x_test.append(aa)
identity=range(0,400)
identity=np.array(identity)

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    CSession = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=CSession)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        print np.shape(x_test)
        # Generate batches for one epoch
        batches = helper.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for XTestB in batches:
            XTestB=np.resize(XTestB,(XTestB.shape[0],14,4,1))
            PredINBatch = sess.run(predictions, {input_x: XTestB, dropout_keep_prob: 1.0})
            print np.shape(PredINBatch)
            all_predictions = (np.concatenate([all_predictions, PredINBatch])).astype(int)


# Save the evaluation to a csv
a=np.array(["id","prediction"])
print np.shape(all_predictions)
FinalPred = np.column_stack((identity, all_predictions))
print np.shape(FinalPred)

FinalPred=np.vstack((a,FinalPred))
print FinalPred
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(FinalPred)
