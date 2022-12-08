# Understanding and Combating Robust Overfitting via Input Loss Landscape Anlaysis and Regularization	

This repository contains code of algorithm AdvLC and pre-trained models from the paper (published in Pattern Recognition 2022) "[Understanding and Combating Robust Overfitting via Input Loss Landscape Anlaysis and Regularization](https://www.sciencedirect.com/science/article/pii/S0031320322007087)". 

# Pre-trained Models

Please find the pre-trained models through this [link](https://emckclac-my.sharepoint.com/:f:/g/personal/k19010102_kcl_ac_uk/EryYoX8PUDdDn1SDv4GCCpwBzksC38_fFB3hnXrzD2SpAQ?e=1F8Ed9).

# Files

* `data`: dataset
* `model`: model checkpoints
  * `trained`: saved model checkpoints
* `output`: experiment logs
* `src`: source code
  * `train.py`: training models
  * `adversary.py`: evaluating adversarial robustness
  * `utils`: shared utilities such training, evaluation, log, printing, adversary, multiprocessing distribution
  * `model`: model architectures
  * `data`: datasets
  * `config`: configurations for training and adversarial evaluation



# Requirements

The development environment is:

1. Python 3.8.13
2. PyTorch 1.11.0 + torchvision 0.12.0

The remaining dependencies are specified in the file `requirements.txt` and can be easily installed via the command:

```p
pip install -r requirements.txt
```

To prepare the involved dataset (pre-trained model), an optional parameter `--download`(`--ptrained`) should be specified in the running command. The program will download the required files automatically. This functionality currently doesn't support the dataset Tiny ImageNet.

# Dependencies

* The training script is based on the PyTorch official [example](https://github.com/pytorch/examples/tree/master/imagenet)
* the code of Wide ResNet is a revised version of [wide-resnet.pytorch](https://github.com/meliketoy/wide-resnet.pytorch).
* the code of PreAct ResNet is from [Alleviate-Robust_Overfitting](https://github.com/VITA-Group/Alleviate-Robust-Overfitting)
* Stochastic Weight Averaging (SWA): [Alleviate-Robust_Overfitting](https://github.com/VITA-Group/Alleviate-Robust-Overfitting)
* Hessian spectrum computation: [PyHessian](https://github.com/amirgholami/PyHessian)

# Training

To adversarially train a PreAct ResNet18 on CIFAR10 using PGD10, run:

```python
python src/train.py -a paresnet --depth 18 --max_iter 10
```

To adversarially train a PreAct ResNet18 on CIFAR10 using FGSM-N, run:

```python
python src/train.py -a paresnet --depth 18 -ri --eps_step 8
```

To adversarially train a PreAct ResNet18 on CIFAR10 using PGD10 with the proposed regularization, run:

```python
python src/train.py -a paresnet --depth 18 --max_iter 10 --reg_lam 0.3 --reg_top 0.1
```

To adversarially train a PreAct ResNet18 on CIFAR10 using PGD10 with the proposed regularization with SWA, run:

```python
python src/train.py -a paresnet --depth 18 --max_iter 10 --reg_lam 0.4 --reg_top 0.1 --swa 50 n 500
```

There are also a lot of hyper-parameters allowed to be specified in the running command in order to control the training. The common hyper-parameters, shared between `src/train.py` and `src/adversary.py` are stored in the `src/config/config.py` and the task-specific hyper-parameters are defined in the corresponding configuration file in the `src/config` folder. Please refer to the specific configuration file for the details of the default and the available options.

# Evaluation

For each training, the checkpoints will be saved in `model/trained/{log}` where {log} is the name of the experiment logbook (by default, is `log`). Each instance of training is tagged with a unique identifier, found in the logbook `output/log/{log}.json`, and that id is later used to load the well-trained model for the evaluation.

To evaluate the robustness of the "best" checkpoint against PGD50, run:

```
python src/adversary.py 0000 -v pgd -a PGD --max_iter 50
```

Similarly against AutoAttack (AA), run:

```
python src/adversary.py 0000 -v pgd -a AA
```

where "0000" should be replaced the real identifier to be evaluated.