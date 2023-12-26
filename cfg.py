gpu = 1

lt = 1.
lt_alpha = 1.
lb = 1.
lb_beta = 10.
lf = 1.
lf_theta_1 = 10.
lf_theta_2 = 1.
lf_theta_3 = 500.
epsilon = 1e-8

# train
learning_rate = 1e-4 
decay_rate = 0.9
beta1 = 0.9
beta2 = 0.999 
max_iter = 700000
show_loss_interval = 50
write_log_interval = 50
save_ckpt_interval = 10000
gen_example_interval = 5000
num_diff_steps = 750
checkpoint_savedir = './checkpoints/eng_hin/'
# ckpt_path = '/DATA/ocr_team_2/onkar/eng_chi_model_logs/train_step-340000.model'
trained_model_path = '/DATA/ocr_team_2/VTNet/checkpoint/larger_vocab_eng_hin/train_step-680000.model' # load from previous checkpoint

# data
batch_size = 46#56
data_shape = [64, None]
# data_dir = 'final_dataset/eng_ger/train'
data_dir = '/DATA/ocr_team_2/onkar/final_dataset/eng_hin/train'
i_t_dir = 'i_t'
i_s_dir = 'i_s'
t_sk_dir = 't_sk'
t_t_dir = 't_t'
t_b_dir = 't_b'
t_f_dir = 't_f'
mask_t_dir = 'mask_t'
example_data_dir = 'bg_large_crops'
example_result_dir = 'gen_logs'

# predict
predict_ckpt_path = None
predict_data_dir = None
predict_result_dir = 'custom_feed/result'
