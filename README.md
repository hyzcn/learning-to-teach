# Implementation of Learning to teach.
For deatils please refer to [Learning to teach](https://openreview.net/pdf?id=HJewuJWCZ).

# Current implementation & TODO
- [x] support cifar10.
- [ ] Reproduce the results on cifar10. (On going, training)



# Dataset:
First download splitted dataset.
```
mkdir data
cd data
wget http://www.cs.toronto.edu/~cqwang/projects/active-learning/data/learning-to-teach/cifar10-splitted.tar.gz
tar -xzvf cifar10-splitted.tar.gz
```

# How to run
```
python train.py --hparams=cifar10_l2t
```


