# Understanding and Combating Robust Overfitting via Input Loss Landscape Anlaysis and Regularization	
# Architecture

* `data`: raw input data
* `model`: model architecture files and the well-trained model checkpoints
  * `architecture`: model architecture files
* `src`: all experiment code
  * `train.py`: main file for training
  * `adversary.py`: main file for evaluating adversarial robustness
  * `utils`: shared utilities such training, evaluation, log, printing, adversary, multiprocessing distribution
  * `model`: networks, Wide ResNet, and input gradient loss
  * `data`: variants of the specific datasets
  * `config`: configurations for training and adversarial evaluation



# Requirements

The development environment is:

1. Python 3.8.13
2. PyTorch 1.11.0 + torchvision 0.12.0



The remaining dependencies are specified in the file `requirements.txt` and can be easily installed via the command:

```p
pip install -r requirements.txt
```



To prepare the involved dataset (pre-trained model), an optional parameter `--download`(`--ptrained`) should be specified in the running command. The program, then, download the required files automatically.

# Dependencies

* The training script is based on the PyTorch official [example](https://github.com/pytorch/examples/tree/master/imagenet)

* the code of Wide ResNet is a revised version of an open-source PyTorch [implementation](https://github.com/meliketoy/wide-resnet.pytorch).
* [PreActResNet](https://github.com/VITA-Group/Alleviate-Robust-Overfitting)
* Hessian spectrum computation: [PyHessian](https://github.com/amirgholami/PyHessian)

# Training

To train a WRN-34 on CIFAR10 without adversarial training, run:

```
python src/train.py -na
```

To adversarially train a WRN-34 on CIFAR10 using PGD10, run:

```
python src/train.py --max_iter 10
```

To adversarially train a WRN-34 on CIFAR10 using PGD10 with the proposed regularization, run:

```
python src/train.py --max_iter 10 --reg advcons 1 l1 top:0.1
```

There are also a lot of hyper-parameters allowed to be specified in the running command in order to control the training. The common hyper-parameters, shared between `src/train.py` and `src/adversary.py` are stored in the `src/config/config.py` and the task-specific hyper-parameters are defined in the corresponding configuration file in the `src/config` folder. Please refer to the specific configuration file for the details of the default and the available options.



For each training, the checkpoints will be saved in `model/log` where "log" is the default name of the experiment logbook and will change as the logbook name. Each instance of training is tagged with a unique identifier, found in the logbook `output/log/log.json`, and that id is later used to load the well-trained model for the evaluation.

To evaluate the FGSM robustness, run:

```
python src/adversary.py 0000
```

To evaluate the PGD20 robustness, run:

```
python src/adversary.py 0000 --attack PGD --max_iter 20
```

where "0000" should be replaced the real identifier to be evaluated.