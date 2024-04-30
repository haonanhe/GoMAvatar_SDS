cd /home/ubuntu/Codes/GoMAvatar

# ZJU-MoCap
SCENE=377
# python -m pdb train.py --cfg exps/zju-mocap_"$SCENE".yaml
python -m pdb train_sds.py --cfg exps/zju-mocap_"$SCENE".yaml