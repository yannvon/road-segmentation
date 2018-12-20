# CS-433 Project 2: Road Segmentation

**This README should allow anyone to replicate our obtained results, for more detailed results,  interpretations, methodology see our report.**

## Introduction

This is a witty introduction to this fantastic subject we are very passionate about.

![nasa-satellite](pictures/nasa-satellite.jpg)


## Project structure

The project is structured as follows:

```
.
├── dataset                 # Location of the compressed dataset
│   ├── train.csv           # The training data set, with labels
│   └── test.csv            # The testing data set, no labels
├── template                # Template files given to us
├── pictures                # Contains some pictures for this README and the report
├── LICENSE
└── README.md
```

> Important: All files and scripts are designed for Python 3.

## Data set

The data set can be downloaded [the EPFL private challenge](https://www.crowdai.org/challenges/epfl-ml-road-segmentation). From the description we see that the data is structured as follows: 

- **test set images**
- **training set**

Our task is to create the following file:

- **submission.csv**



## Performance of our models

### Basic model with window of 80 pixels

All 200 epochs (with Early stop)

| Model Description                                            | Training Accuracy | crowdAI F1 |
| ------------------------------------------------------------ | ----------------- | ---------- |
| 80 window Basic Model, no dense, 0.1 dropout, 10% rotations  | 0.972             | 0.881      |
| 80 window Basic Model, no dense, 0.1 dropout, optimized rotations | 0.963             | 0.871      |
| 80 window Basic Model, no dense, 0.2 dropout, optimized rotations | 0.955             | 0.843      |

### Varying window sizes on basic model

All 200 epochs (With Early stop)

| Model Description                                            | Training Accuracy | crowdAI F1 |
| ------------------------------------------------------------ | ----------------- | ---------- |
| 64 window Basic Model, no dense, 0.2 dropout, full rotations | 0.950             | -          |
| 80 window Basic Model, no dense, 0.2 dropout, full rotations | 0.955             | 0.843      |
| 100 window Basic Model, no dense, 0.2 dropout, full rotations | 0.963             | 0.870      |
| 120 window Basic Model, no dense, 0.2 dropout, full rotations | 0.949             | 0.826      |

### Monday afternoon experiments

All 200 epochs (With Early stop)

| Model Description                                            | Training Accuracy | crowdAI F1 |
| ------------------------------------------------------------ | ----------------- | ---------- |
| 80 window Basic Model, no dense, 0.1 dropout, augmented data | diverges (0.75)   | -          |
| 80 window Basic Model, no dense, 0.1 dropout, optimized rotations | 0.970             | 0.862      |

### Monday night experiments

All 200 epochs (With Early stop)

All with 0.2 validation

| id   | Model Description                                            | Training Accuracy | Validation Accuracy | crowdAI F1 | comments                                 |
| ---- | ------------------------------------------------------------ | ----------------- | ------------------- | ---------- | ---------------------------------------- |
| 1    | 80 window Basic Model, no dense, 0.1 dropout, 10% rotation   | 0.961             | 0.922               | 0.859      | without validation achieved 0.881 ? How? |
| 2    | 100 window Basic Model, no dense, 0.1 dropout, 10% rotation  | 0.966             | 0.919               | 0.856      | ????                                     |
| 3    | 100 window Basic Model, no dense, 0.25 dropout, 10% rotation, initial learning rate = 0.001 | 0.965             | 0.920               |            |                                          |
| 4    | 100 window Basic Model, 1 dense, 0.1 dropout, 10% rotation, initial learning rate = 0.0005 | 0.976             | 0.944               |            | needs rerun                              |
| 5    | 100 window Basic Model, 1 dense, 0.25 dropout, basic rotation, initial learning rate = 0.0005 | 0.965             | 0.931               |            |                                          |
| 6    | 100 window Basic Model, no dense, 0.1 dropout, basic rotation, 50/50 data, initial learning rate = 0.001 | 0.271             | 0.257               | -          | diverged !                               |
| 6    | 100 window Basic Model, no dense, 0.1 dropout, basic rotation, 50/50 data, initial learning rate = 0.0005 | 0.980             | 0.943               | 0.884      | new all time best!                       |

### Tuesday experiments

100 window Basic Model, no dense, 0.1 dropout, basic rotation, 50/50 data, initial learning rate = 0.0005

Tiago Bugfix padding

All 200 epochs (With Early stop)

All with 0.2 validation

| id   | Model Description                | Training Accuracy | Validation Accuracy | crowdAI F1 | comments |
| ---- | -------------------------------- | ----------------- | ------------------- | ---------- | -------- |
| 1    | same model                       | 0.978             | 0.941               | 0.884      |          |
| 2    | 1 max pooling less (between 128) | 0.987             | 0.940               | 0.88       |          |
| 3    | 0.25 droupout                    | 0.978             | 0.933               | 0.869      |          |
| 4    | dense                            | 0.978             | 0.942               | 0.881      |          |
| 5    | augmented                        |                   |                     | 0.869      |          |
| 6    | regularizer l2                   |                   |                     |            |          |



### Tuesday night experiments

100 window Basic Model, no dense, 0.1 dropout, basic rotation, 50/50 data, initial learning rate = 0.0005

Tiago Bugfix padding

All 200 epochs (With Early stop)

All with 0.2 validation



| id   | Model Description                | Training Accuracy | Validation Accuracy | crowdAI F1 | comments                    |
| ---- | -------------------------------- | ----------------- | ------------------- | ---------- | --------------------------- |
| 1    | threshold = 0.2                  | 0.981             | 0.944               | 0.861      |                             |
| 2    | threshold = 0.25                 | 0.975             | 0.941               |            |                             |
| 3    | threshold = 0.3                  | 0.976             | 0.941               |            | no difference apparently .. |
| 4    | threshold = 0.35                 | 0.981             | 0.941               |            |                             |
| 5    | threshold = 0.25 and up-sampling | 0.989             | 0.940               |            |                             |

### Other tests

- Relu vs LeakyRelu
- kernel size
- transfer learning
- Tiago strat



## How to run Keras (with GPU) on a Jupyter notebook from a EC2 (AWS) instance

1. Sign-up for a AWS Educate account, or even better use the promo from the Github student pack to create a full-fledged AWS account.
2. Go to your AWS workbench and follow [this guide](https://hackernoon.com/keras-with-gpu-on-amazon-ec2-a-step-by-step-instruction-4f90364e49ac) until the end of section 3. Note that you should select "Deep Learning AMI (Ubuntu) Version 19.0 " for example from the list of community AMI's.
3. Follow [this AWS guide](https://docs.aws.amazon.com/dlami/latest/devguide/setup-jupyter.html) to run a Jupyter notebook server and access it from your browser. Note that if you have not configured SSL you should use http to access your notebook.



Here are some other tips that might be useful to you :

- The AMI that we installed come with many virtual environments, once you ssh to the machine you can activate the environment you want as described in the login message. Note that to install packages you can use pip, once you activated your virtualenv.
- You will need to tell your Jupyter Notebook which environment to use, this setting can be changed under the `Kernels` tab.

### Useful commands

```python
ssh -i "aws_key.pem" ubuntu@ec2-18-204-43-147.compute-1.amazonaws.com

ssh -i "aws_key.pem" -L 8157:127.0.0.1:8888 ubuntu@ec2-18-204-43-147.compute-1.amazonaws.com

http://127.0.0.1:8157 # Not HTTPS !

source activate tensorflow_p36

pip install opencv-python

git clone https://github.com/yannvon/road-segmentation.git

```



## How to run TensorBoard on an EC2 instance

We use TensorBoard to obtain direct feedback and plots from the progress of the training of the model.

All we have to do is add a callback to the keras fit method, as well as running a TensorBoard server on the instance. The following commands are useful:

### Useful commands

```
 ssh -i "aws_key.pem" -L 16006:127.0.0.1:6006 ubuntu@ec2-34-206-1-189.compute-1.amazonaws.com

tensorboard --logdir=/home/ubuntu/road-segmentation/src/logs

http://127.0.0.1:16006
```

## Alternative way to train the model

```bash
(tensorflow_p36) $ nohup python run.py > /dev/null 2>&1&
```

## Kaggle Kernels

K80 for free during 6h session ! Can run many sessions at the same time !





## Our Approach

Non deterministic if trained on GPU !



## Team members

- Benno Schneeberger
- Tiago Kieliger
- Yann Vonlanthen

## Report Notes



## Further ideas

- [x] Use pre- trained models

