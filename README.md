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

tensorbod --logdir=/home/ubuntu/road-segmentation/src/logs

http://127.0.0.1:16006
```

## Alternative way to train the model

```bash
(tensorflow_p36) $ nohup python run.py > /dev/null 2>&1&
```



## Our Approach





## Team members

- Benno Schneeberger
- Tiago Kieliger
- Yann Vonlanthen

## Report Notes



1. No cross validation -justification
2. Illustrations to show our claim
3. Always justify with sources
4. Display filters would be nice
5. Drop out, to avoid overfit - add some tables
6. Data augmentation (rotation of 45 or 90 deg? chosen at random?)
7. Baseline (without doing anything? using a method seen previously?)
8. Handle Boundaries (mirror, which size?)
9. Activation function (test which one is the best? relu, relu leak,..)
10. Take a bigger to evaluate the center (size of the window? select all possible windows or only a few? How many?)
11. Regularization? 
12. Make sure same amount of road and non road training data - done in template
13. Sliding window to have much more data.

## Further ideas

- [ ] Use pre- trained models

