date

#### AimCLR NTU-60 xsub ####

# Pretext
python main.py pretrain_aimclr --config config/ntu60/pretext/pretext_aimclr_xsub_joint.yaml
python main.py pretrain_aimclr --config config/ntu60/pretext/pretext_aimclr_xsub_motion.yaml
python main.py pretrain_aimclr --config config/ntu60/pretext/pretext_aimclr_xsub_bone.yaml

# Linear_eval
python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_aimclr_xsub_joint.yaml
python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_aimclr_xsub_motion.yaml
python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_aimclr_xsub_bone.yaml

# Ensemble
python ensemble_ntu_cs.py