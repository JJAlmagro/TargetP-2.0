import numpy as np

def precision_cs(sp_cs, mt_cs, ch_cs, th_cs, pred_type, y_type, y_cs):
	'''Calculate precision for the cleavage site
	'''
	correct_sp = sp_cs == y_cs
	correct_mt = mt_cs == y_cs
	correct_ch = ch_cs == y_cs
	correct_th = th_cs == y_cs

	# Precision
	prec_sp = np.sum(correct_sp[np.where((pred_type == 1)&(y_type == 1))])/np.sum(pred_type == 1)
	prec_mt = np.sum(correct_mt[np.where((pred_type == 2)&(y_type == 2))])/np.sum(pred_type == 2)
	prec_ch = np.sum(correct_ch[np.where((pred_type == 3)&(y_type == 3))])/np.sum(pred_type == 3)
	prec_th = np.sum(correct_th[np.where((pred_type == 4)&(y_type == 4))])/np.sum(pred_type == 4)

	return prec_sp, prec_mt, prec_ch, prec_th


def recall_cs(sp_cs, mt_cs, ch_cs, th_cs, pred_type, y_type, y_cs):
	'''Calculate recall for the cleavage site
	'''
	correct_sp = sp_cs == y_cs
	correct_mt = mt_cs == y_cs
	correct_ch = ch_cs == y_cs
	correct_th = th_cs == y_cs

	correct_tp = pred_type == y_type

	correct_both_sp = correct_sp&correct_tp
	correct_both_mt = correct_mt&correct_tp
	correct_both_ch = correct_ch&correct_tp
	correct_both_th = correct_th&correct_tp

	# Recall
	recall_sp = np.mean(correct_both_sp[np.where(y_type == 1)])
	recall_mt = np.mean(correct_both_mt[np.where(y_type == 2)])
	recall_ch = np.mean(correct_both_ch[np.where(y_type == 3)])
	recall_th = np.mean(correct_both_th[np.where(y_type == 4)])

	return recall_sp, recall_mt, recall_ch, recall_th

