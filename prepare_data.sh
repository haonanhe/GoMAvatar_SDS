cd /home/ubuntu/Codes/GoMAvatar

cd scripts/prepare_zju-mocap

SCENE=377
# python prepare_dataset.py --cfg "$SCENE".yaml
python -m pdb prepare_dataset_smplx.py --cfg "$SCENE".yaml