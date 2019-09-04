import numpy as np
import sys
import os
import argparse
from metrics import *
import sklearn.metrics as metrics

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str,  
	help="npz file with TargetP input data in BLOSUM62")
parser.add_argument('-g','--gpu', default=-1, type=int, 
	help='the GPU number, -1 indicates CPU')
parser.add_argument('-m','--model', type=str, 
	help='folder where the models have been saved')
args = parser.parse_args()


gpu = args.gpu 

if gpu > -1:
	device = '/gpu:%i' % gpu 
	os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
else:
	device = '/cpu:0'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if args.dataset == None:
	parser.print_help()
	sys.stderr.write("Please specify the dataset npz file!\n")
	sys.exit(1)

if args.model == None:
	parser.print_help()
	sys.stderr.write("Please specify the folder where the models have been saved!\n")
	sys.exit(1)

import tensorflow as tf

# Load data
print('Loading dataset...')
data = np.load(args.dataset)

folds = data['fold']

# Initialize list to hold the predictions
all_preds = []
all_sp = []
all_mt = []
all_ch = []
all_th = []
all_tp = []
all_y = []
all_ids = []

# Nested cross-validation prediction
partitions_interval = np.arange(5)
for test_partition in partitions_interval:
	inner_partitions_interval = partitions_interval[partitions_interval != test_partition]

	# Load the test set of each fold
	test_set = np.where(folds == test_partition)
	X_test = data['x'][test_set]
	y_test = data['y_cs'][test_set]
	len_test = data['len_seq'][test_set]
	org_test = data['org'][test_set]
	tp_test = data['y_type'][test_set]
	ids_test = data['ids'][test_set]

	n_seq = X_test.shape[0]
	n_steps = X_test.shape[1]
	n_feat = X_test.shape[2]
	batch_size = 64
	n_type = 5

	if n_seq % batch_size == 0:
		n_batch = int(n_seq/batch_size)
	else:
		n_batch = int(n_seq/float(batch_size)) + 1

	test_cs_sp = np.zeros((n_seq,n_steps))
	test_cs_mt = np.zeros((n_seq,n_steps))
	test_cs_ch = np.zeros((n_seq,n_steps))
	test_cs_th = np.zeros((n_seq,n_steps))
	test_prot_type = np.zeros((n_seq,n_type))

	print('Testing on partition: %i' % test_partition)
	print('=' * 30)

	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.per_process_gpu_memory_fraction = 0.4
	# The 4 models of the inner cross-validaton perform a prediction on the test set
	for val_partition in inner_partitions_interval:
		# Reset TF graph
		tf.reset_default_graph()
		with tf.device(device):
			# Initialize session and load model parameters
			sess=tf.Session(config=config)
			model_file = "%s/partition_%i_%i/model.meta" % (args.model, test_partition,val_partition)
			saver = tf.train.import_meta_graph(model_file, clear_devices=True)
			saver.restore(sess,tf.train.latest_checkpoint('%s/partition_%i_%i/' % (args.model,
				test_partition,val_partition)))
			
			graph = tf.get_default_graph()

			# Extract input tensors
			x = graph.get_tensor_by_name('Input_protein:0')
			seq_len = graph.get_tensor_by_name('Protein_length:0')
			organism = graph.get_tensor_by_name('Organism:0')
			is_training_pl = graph.get_tensor_by_name('Training:0')
			r_drop = graph.get_tensor_by_name('RNN_prob:0')

			# Extract output tensors
			cs_sp = graph.get_tensor_by_name('SP_CS:0')
			cs_mt = graph.get_tensor_by_name('MT_CS:0')
			cs_ch = graph.get_tensor_by_name('CH_CS:0')
			cs_th = graph.get_tensor_by_name('TH_CS:0')
			type_pred = graph.get_tensor_by_name('Protein_pred:0')

			# Get prediction
			test_ind = np.arange(n_seq)
			batch_lst = np.arange(n_batch)

			# Iterate through batch list
			for batch_i in batch_lst:
				pr_batch = batch_i*batch_size
				batch_inds = test_ind[pr_batch:pr_batch+batch_size]

				feed_dict = {x: X_test[batch_inds], seq_len:len_test[batch_inds], organism:org_test[batch_inds], r_drop: 1.0, is_training_pl:False}
				pred_cs_sp, pred_cs_mt, pred_cs_ch, pred_cs_th, pred_prot= sess.run(
					[cs_sp, cs_mt, cs_ch, cs_th, type_pred], feed_dict)

				# Sum prediction for each of the 4 models in the inner cross-validation
				test_cs_sp[batch_inds] += pred_cs_sp
				test_cs_mt[batch_inds] += pred_cs_mt
				test_cs_ch[batch_inds] += pred_cs_ch
				test_cs_th[batch_inds] += pred_cs_th

				test_prot_type[batch_inds] += pred_prot

	# Average the prediction
	test_sp_avg = test_cs_sp/4.0
	test_mt_avg = test_cs_mt/4.0
	test_ch_avg = test_cs_ch/4.0
	test_th_avg = test_cs_th/4.0
	test_type_avg = test_prot_type/4.0

	# Save the prediction to a list
	all_sp.append(test_sp_avg)
	all_mt.append(test_mt_avg)
	all_ch.append(test_ch_avg)
	all_th.append(test_th_avg)
	all_preds.append(test_type_avg)
	all_tp.append(tp_test)
	all_y.append(y_test)
	all_ids.append(ids_test)


# Concatenate all the predictions
all_preds_conc = np.concatenate(all_preds,axis=0)
all_sp_conc = np.concatenate(all_sp,axis=0)
all_mt_conc = np.concatenate(all_mt,axis=0)
all_ch_conc = np.concatenate(all_ch,axis=0)
all_th_conc = np.concatenate(all_th,axis=0)
all_y_conc = np.concatenate(all_y,axis=0)
all_ids_conc = np.concatenate(all_ids,axis=0)

# Argmax to get predicted label
arg_type = np.argmax(all_preds_conc, 1)
arg_sp = np.argmax(all_sp_conc, 1)
arg_mt = np.argmax(all_mt_conc, 1)
arg_ch = np.argmax(all_ch_conc, 1)
arg_th = np.argmax(all_th_conc, 1)
arg_y = np.argmax(all_y_conc, 1)
all_tp = np.concatenate(all_tp,axis=0)


# Calculate precision and recall for the cleavage site
prec_sp, prec_mt, prec_ch, prec_th = precision_cs(arg_sp, arg_mt,arg_ch, arg_th, arg_type, all_tp, arg_y)
recall_sp, recall_mt, recall_ch, recall_th = recall_cs(arg_sp, arg_mt,arg_ch, arg_th, arg_type, all_tp, arg_y)

# Calculate f1 score
f1_type = metrics.f1_score(all_tp, arg_type, average=None)

# Summary
print('========== Final results ==========')
print('Signal\tF1 score\tPrec. CS\tRec. CS')
print('noTP\t%.6f\t%.6f\t%.6f' % (f1_type[0], 0.0, 0.0))
print('SP\t%.6f\t%.6f\t%.6f' % (f1_type[1], prec_sp, recall_sp))
print('mTP\t%.6f\t%.6f\t%.6f' % (f1_type[2], prec_mt, recall_mt))
print('cTP\t%.6f\t%.6f\t%.6f' % (f1_type[3], prec_ch, recall_ch))
print('luTP\t%.6f\t%.6f\t%.6f' % (f1_type[4], prec_th, recall_th))
