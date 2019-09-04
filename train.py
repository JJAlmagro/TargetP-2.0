
import numpy as np
import os
import sys
import time
import argparse
import sklearn.metrics as metrics


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str,  
	help="npz file with TargetP input data in BLOSUM62")
parser.add_argument('-g','--gpu', default=-1, type=int, 
	help='the GPU number, -1 indicates CPU')
parser.add_argument('-p','--prefix', default='targetp_train', type=str, 
	help='prefix of the output folder where the models are saved')
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

import tensorflow as tf
from model import network

# Load dataset and create output folder
print('Loading dataset...')
data = np.load(args.dataset)
timestr = time.strftime("%Y%m%d-%H%M%S")
logdir = '%s_%s/' % (args.prefix, timestr)
os.mkdir(logdir)

folds = data['fold']

print('Training...')
total_models = 0
# Nested cross-validation loop with 5 folds
partitions_interval = np.arange(5)
for test_partition in partitions_interval:
	inner_partitions_interval = partitions_interval[partitions_interval != test_partition]
	# Inner cross-validation
	for val_partition in inner_partitions_interval:
		# Create directory to store the model
		os.mkdir('%spartition_%i_%i' % (logdir, test_partition, val_partition))
		model_file = "%spartition_%i_%i/model" % (logdir, test_partition, val_partition)

		# Define train and validation splits
		train_partition = inner_partitions_interval[inner_partitions_interval != val_partition]
		train_set = np.in1d(folds.ravel(), train_partition).reshape(folds.shape)
		val_set = np.where(folds == val_partition)




		# Load training data
		X_train = data['x'][train_set]
		y_train = data['y_cs'][train_set]
		len_train = data['len_seq'][train_set]
		org_train = data['org'][train_set]
		tp_train = data['y_type'][train_set]

		# Load validation data
		X_val = data['x'][val_set]
		y_val = data['y_cs'][val_set]
		len_val = data['len_seq'][val_set]
		org_val = data['org'][val_set]
		tp_val = data['y_type'][val_set]

		# Network Parameters
		n_input = 20
		n_steps = 200
		training_iters = 150
		batch_size = 64

		# Hyperparameters
		learning_rate = 0.002
		n_hidden_rnn = 256
		n_filt = 32
		n_hidden = 256
		n_att = 13
		att_size = 144
		e_drop = 0.6
		i_drop = 0.75
		rnn_drop = 0.5
		filt_size = 1
		gamma = 0.1

		# Set seed
		np.random.seed(val_partition)
		tf.reset_default_graph()
		tf.set_random_seed(val_partition)

		# Set device
		with tf.device(device):
			# Initialize network
			x, y, seq_len, is_training_pl, r_drop, lr, organism, type_prot, train_adam, \
			loss, type_pred, sp_prob, mt_prob, ch_prob, th_prob = network(n_hidden_rnn, n_filt, 
				n_hidden, filt_size, i_drop, e_drop, n_att, att_size)

			# Initialize saver
			saver = tf.train.Saver()

			# Initializing the variables
			init = tf.global_variables_initializer()
			init_l = tf.local_variables_initializer()
			config = tf.ConfigProto(allow_soft_placement=True)
			config.gpu_options.per_process_gpu_memory_fraction = 0.4

			total_models += 1
			print('=' * 98)
			print('Outer (left out) partition: %i | Inner (validation) partition: %i ---- Model %i/20' % (test_partition,
			 val_partition, total_models))
			print('=' * 98)
			# Start session
			with tf.Session(config=config) as sess:
				sess.run(init)
				sess.run(init_l)

				# Keep track of the lowest loss
				fold_val = float('inf')

				# Create batches for validation
				n_seq_val = X_val.shape[0] 
				if n_seq_val % batch_size == 0:
					n_batch_val = int(n_seq_val/batch_size)
				else:
					n_batch_val = int(n_seq_val/float(batch_size)) + 1

				# Create batches for training
				n_seq = X_train.shape[0] 
				if n_seq % batch_size == 0:
					n_batch_tr = int(n_seq/batch_size)
				else:
					n_batch_tr = int(n_seq/float(batch_size)) + 1
				
				best_val_epoch = []

				lr_red = 0

				# Training loop
				for epoch in range(training_iters):

					start = time.time()

					# Reduce learning rate if there is no improvement after 8 epochs
					if epoch > 10:
						if (epoch - best_val_epoch[-1]) == 8:
							learning_rate *= gamma
							best_val_epoch.append(epoch)
							lr_red += 1

					# Stop training after learning rate has been reduced too many times
					if lr_red == 5:
						break
						print('Stopping training early')

					# Create batch list and shuffle examples		
					loss_train = []
					train_ind = np.arange(n_seq)
					np.random.shuffle(train_ind)
					batch_lst = np.arange(n_batch_tr)

					tr_pred_batch = []
					tr_type_batch = []

					# Iterate through batch list
					for batch_i in batch_lst:
						pr_batch = batch_i*batch_size
						batch_inds = train_ind[pr_batch:pr_batch+batch_size]

						# Training and report loss
						tr_loss, tr_pred, _ = sess.run([loss, type_pred, train_adam], 
						feed_dict={x: X_train[batch_inds], y: y_train[batch_inds], seq_len: len_train[batch_inds], lr: learning_rate, 
							organism: org_train[batch_inds], type_prot: tp_train[batch_inds], r_drop: rnn_drop, is_training_pl:True})

						loss_train.append(tr_loss)
						tr_pred_batch.extend(np.argmax(tr_pred,1))
						tr_type_batch.extend(tp_train[batch_inds])

					f1_train = metrics.f1_score(tr_type_batch, tr_pred_batch, average='macro')

					# Create batch list
					loss_val = []
					val_ind = np.arange(n_seq_val)
					batch_lst_val = np.arange(n_batch_val)

					val_pred_batch = []
					val_type_batch = []


					# Iterate through batch list
					for batch_i in batch_lst_val:
						pr_batch = batch_i*batch_size
						batch_inds = val_ind[pr_batch:pr_batch+batch_size]

						# Calculate validation loss
						vl_loss, vl_pred = sess.run([loss,type_pred], feed_dict={x: X_val[batch_inds], seq_len:len_val[batch_inds], y: y_val[batch_inds],
							organism:org_val[batch_inds], type_prot:tp_val[batch_inds], is_training_pl:False, r_drop: 1.0})
						
						loss_val.append(vl_loss)
						val_pred_batch.extend(np.argmax(vl_pred,1))
						val_type_batch.extend(tp_val[batch_inds])


					f1_val = metrics.f1_score(val_type_batch, val_pred_batch, average='macro')

					# Epoch loss
					training_loss = np.mean(loss_train)
					validation_loss = np.mean(loss_val)


					now = time.time() - start
					print('-' * 98)
					print('Epoch: %i | %.1f s | train loss: %.4f | train f1: %.4f | valid loss: %.4f | valid f1: %.4f' % (epoch+1, 
						now ,training_loss, f1_train, validation_loss, f1_val))

					# Save model if the validation loss decreases
					if fold_val > validation_loss:
						fold_val = validation_loss
						best_val_epoch.append(epoch)
						saver.save(sess, model_file)

print('End of training')

