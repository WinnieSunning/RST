## RST: Rough set Transformer for 3D Point Cloud Learning
This is a Pytorch implementation of RST

### Requirements
python >= 3.6

pytorch >= 1.6

h5py

scikit-learn

and

```shell script
pip install pointnet2_ops_lib/.
python install setup.py
```
Modified from PCT  https://github.com/Strawberry-Eat-Mango/PCT_Pytorch.

### Models
We provide two models to train, our model is placed in RSTmodel.py

And If you want to reproduce PCT, please choose PCTmodel.py 

### Example training and testing
```shell script
# train
python main.py --exp_name=train --num_points=1024 --use_sgd=True --batch_size 32 --epochs 600 --lr 0.0001

# test
python main.py --exp_name=test --num_points=1024 --use_sgd=True --eval=True --model_path=checkpoints/train/models/model.t7 --test_batch_size 8

```



