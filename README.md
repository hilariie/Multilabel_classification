# One vs All Multilabel_classification
This project aims to simplify the implementation of a multiclass problem using the one-vs-all (OvA) multilabel approach. 
Although this approach is effective in producing high-accuracy models, it can be tedious to implement, particularly when
dealing with a problem with many classes. This project provides a set of functions and classes to make the 
implementation process easy to use.

The project is designed to work with any pre-trained state-of-the-art models on Keras (e.g VGG16, ResNet, etc.) by 
simply modifying the [configuration file](config.yaml)

## What is One-vs-All?
OvA classification is a technique used in multi-class classification that extends binary classifiers to handle
multi-class problems. The basic idea behind OvA is to train multiple binary classifiers, where each classifier
is responsible for distinguishing one specific class from the rest of the classes.

In the OvA (One-vs-All) approach, we treat a multiclass problem as multiple binary classification tasks. 
For example, if we have a classification task to identify images of cats, dogs, and birds, we would train three binary classifiers. 
The first binary classifier would determine if an image is a cat or not, the second for dogs, and the third for birds. 
The final layer of the neural network would have all three binary classifiers instead of the normal approach 
(softmax regression), which would have a single multilabel classifier.

Using multiple binary classifiers surprisingly does not increase training time, and it significantly 
outperforms the single multilabel classifier in the examples used.

The architecture developed in this project was used on two multi-class problems. You can check the inference scripts 
([script1](inference.ipynb), [script2](inference%20II.ipynb)) provided to see the performance on test images.

## OvA approach for image classification
Considering the OvA approach is different from the normal classification approach (softmax regression), developing
models using the OvA approach requires some modifications/changes to the norm. Here are the main steps involved:
1. Data preparation: Given a multi-class dataset with N classes, N binary classifiers are created, one for each class.
For any given class (i), the positive examples include the samples belonging to class i, and the negative examples
include all other samples belonging to the other classes.
2. Training: The Neural network will have N binary classifiers at its final layer.
3. Prediction: When making predictions for an image, all N binary classifiers are used. Each classifier predicts
whether the image belongs to its corresponding class or not. The final class assignment is determined be selecting
the class associated with the classifier that gives the highest confidence or probability score.

## Data Generators
In the OvA approach, y labels are created for each class. As a result, each image has multiple y labels associated with 
it (For example in a case of classifying dogs, cats, and birds, a cat image will have y = [0, 1, 0]. Here, y=0 is used to train
the dog and bird binary classifiers for that image, while y=1 is used to train the cat classifier).

This makes it challenging to use TensorFlow's data generator function. To address this, a custom data generator is 
created in this project that also includes data augmentation.

This custom data generator can be used with minimal or no modification to suit your use case. It assumes that your image
datasets are split into train and test and that they are in their respective directories (i.e., N number of folders, 
which corresponds to the number of classes).

## Callbacks
Using the OvA approach, multiple binary models are trained in parallel. For this reason, it is difficult to use 
TensorFlow's EarlyStopping class while monitoring "val_acc" as there is no "val_acc". Instead, you have 
"val_model1_acc", "val_model2_acc", etc. 

To solve this, a custom callback class was developed to monitor the average 'val_acc' across the models. This class can 
be used with minimal or no modification to suit your use case.

## How to use
To use this architecture on a multiclass problem, please follow the steps below:

1. Clone the repository using `git clone https://github.com/hilariie/Multilabel_classification.git`
2. Navigate to the directory using `cd Multilabel_classification`
3. Modify the `train_path` in the `config.yaml` file to your own training directory.
4. Modify the `val_path` in the `config.yaml` file to your validation directory.
5. Run the main script using `python main.py`
