import tensorflow as tf

def network(n_hidden_rnn, n_filt, n_hidden, filt_size, i_drop, e_drop, n_att, att_size,
	n_input=20, n_steps=200, n_classes=3, n_type=5, n_org=2):
	
		
	# Input variable 
	x = tf.placeholder(tf.float32, [None, n_steps, n_input], name='Input_protein')
	# Label variable
	y = tf.placeholder(tf.int32, [None, n_steps], name='Label_protein')
	# Sequence length variable
	seq_len = tf.placeholder(tf.int32, [None], name='Protein_length')
	# Training variable
	is_training_pl = tf.placeholder(tf.bool, name='Training')
	# Type lable variable
	type_prot = tf.placeholder(tf.int32, [None], name='Protein_type')
	# Organism variable
	organism = tf.placeholder(tf.int32, [None], name='Organism')
	# Learning rate variable
	lr = tf.placeholder(tf.float32, shape=(), name="Learning_rate")
	# Dropout keep probability variable
	r_drop = tf.placeholder(tf.float32, shape=(), name="RNN_prob")

	# One-hot encoding organism
	org_hot = tf.one_hot(organism, n_org, dtype=tf.float32)

	# Input dropout
	x_drop = tf.contrib.layers.dropout(x, noise_shape=[tf.shape(x)[0], n_steps, 1], 
		keep_prob=i_drop, is_training=is_training_pl)

	# Convolutional layer filter size 1
	conv_l1 = tf.layers.conv1d(x_drop, filters=n_filt, kernel_size=filt_size, 
			strides=1, padding="same", activation=tf.nn.elu)
	
	# Dropout convolutional layer
	conv_drop = tf.contrib.layers.dropout(conv_l1, keep_prob=e_drop, is_training=is_training_pl)

	# Dense layer organism type
	l_organism = tf.contrib.layers.fully_connected(org_hot, n_hidden_rnn)

	def lstm_cell(n_h, k_p, name):
		'''LSTM cell function with dropout 
		'''
		cell = tf.contrib.rnn.LSTMCell(n_h, name=name)
		cell = tf.contrib.rnn.DropoutWrapper(cell, 
			state_keep_prob=k_p, output_keep_prob=k_p)
		return cell

	# Forward direction cell
	lstm_fw_cell = lstm_cell(n_hidden_rnn, r_drop, "fw1")#,layer_norm=is_training_pl)
	# Backward direction cell
	lstm_bw_cell = lstm_cell(n_hidden_rnn, r_drop, "bw1")#,layer_norm=is_training_pl)

	# Initial hidden state with organism information
	l_org_cell = tf.contrib.rnn.LSTMStateTuple(l_organism, l_organism)

	# Bidirectional LSTM
	outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputs=conv_drop, 
		initial_state_fw=l_org_cell, initial_state_bw=l_org_cell, 
		sequence_length=seq_len, dtype=tf.float32)

	# Concatenate hidden states bidirectional LSTM
	final_outputs = tf.concat(outputs,2)

	# Multi-attention layer

	# Attention matrix, filter size 1 convolution, equivalent to dense layer.
	hUa = tf.layers.conv1d(final_outputs, filters=att_size, kernel_size=1, 
		strides=1, padding="same", activation=tf.nn.tanh) #, reuse=False)

	# Align matrix, filter size 1 convolution, equivalent to dense layer.
	align = tf.layers.conv1d(hUa, filters=n_att, kernel_size=1, 
		strides=1, padding="same", activation=None)#name='Conv_aln', reuse=False)


	# Mask for padded positions
	masks = tf.expand_dims(tf.sequence_mask(seq_len, dtype=tf.float32, maxlen=n_steps), axis = 2)
	a_un_masked = align * masks - (1 - masks) * 100000

	# Extract attention vector used to calculate CS
	out_sp = a_un_masked[:,:,0]
	out_mt = a_un_masked[:,:,1]
	out_ch = a_un_masked[:,:,2]
	out_th = a_un_masked[:,:,3]

	# Softmax operation
	sp_prob = tf.nn.softmax(out_sp, name='SP_CS')
	mt_prob = tf.nn.softmax(out_mt, name='MT_CS')
	ch_prob = tf.nn.softmax(out_ch, name='CH_CS')
	th_prob = tf.nn.softmax(out_th, name='TH_CS')


	# Generate weight vector to mask loss for the CS site prediction
	ones = tf.ones(tf.shape(type_prot))
	zeros = tf.zeros(tf.shape(type_prot))
	# SP mask
	cond_sp = tf.equal(type_prot,1)
	w_sp = tf.where(cond_sp,ones,zeros)
	# MT mask
	cond_mt = tf.equal(type_prot,2)
	w_mt = tf.where(cond_mt,ones,zeros)
	# CH mask
	cond_ch = tf.equal(type_prot,3)
	w_ch = tf.where(cond_ch,ones,zeros)
	# TH mask
	cond_th = tf.equal(type_prot,4)
	w_th = tf.where(cond_th,ones,zeros)

	# Calculate attention values
	alpha = tf.nn.softmax(a_un_masked, axis=1)

	# Weighted hidden states
	weighted_hidden = tf.expand_dims(final_outputs,2) * tf.expand_dims(alpha, 3)

	# Weighted sum
	weighted_sum = tf.reduce_sum(weighted_hidden, axis=1)

	# Final dense layer
	# weighted_out = tf.squeeze(tf.layers.conv1d(weighted_sum, filters=n_filt2, kernel_size=n_att, 
	# 		strides=1, padding="valid", activation=tf.nn.elu),1)
	weighted_out = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(weighted_sum), n_hidden, activation_fn=tf.nn.elu)

	# Protein type prediction
	type_pred_layer = tf.contrib.layers.fully_connected(weighted_out, n_type, activation_fn=None)
		
	# Softmax operation
	type_pred = tf.nn.softmax(type_pred_layer, name='Protein_pred')
	# Calculate cross-entropy for protein type prediction
	loss_type = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=type_pred_layer, 
		labels=type_prot))

	# Calculate cross-entropy for CS prediction
	loss_sp = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=out_sp, onehot_labels=y, 
		reduction=tf.losses.Reduction.NONE, weights=w_sp))
	loss_mt = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=out_mt, onehot_labels=y, 
		reduction=tf.losses.Reduction.NONE, weights=w_mt))
	loss_ch = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=out_ch, onehot_labels=y, 
		reduction=tf.losses.Reduction.NONE, weights=w_ch))
	loss_th = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=out_th, onehot_labels=y, 
		reduction=tf.losses.Reduction.NONE, weights=w_th))

	# Combined loss
	loss = loss_type + loss_sp + loss_mt + loss_ch + loss_th

	# Training operation
	train_adam = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

	return x, y, seq_len, is_training_pl, r_drop, lr, organism, type_prot, train_adam, \
		loss, type_pred, sp_prob, mt_prob, ch_prob, th_prob
