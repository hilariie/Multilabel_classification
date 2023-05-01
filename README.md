# One vs All Multilabel_classification
This project aims to simplify the implementation of a multiclass problem using the one vs all (OvA) multilabel approach. Although this approach is effective in producing high-accuracy models, it can be tedious to implement, particularly when dealing with a problem with many classes. This project provides a set of functions to make the implementation process more straightforward.

## What is One vs All?
In OvA, we treat a multiclass problem as multiple binary classification tasks. For example, if we have a classification task to identify images of cats, dogs, and birds, we would train three binary classifiers. The first binary classifier would determine if an image is a cat or not, the second for dogs, and the third for birds. The final layer of the neural network would have three binary classifiers instead of the normal approach, which would have only one multilabel classifier.

Using multiple binary classifiers surprisingly does not increase training time, and it manages to significantly outperform the single multilabel prediction layer.

The architecture developed in this project was used for a multiclass problem of identifying local Nigerian meals. An inference script is provided to show the output on test images.


## Data Generators
In the OvA approach, y labels are created for each class. As a result, each image has multiple y labels associated with it. This makes it challenging to use TensorFlow's data generator function. To address this, a custom data generator is created in this project that also includes data augmentation.

This custom data generator can be used with minimal or no modification to suit your use case. It assumes that your image datasets are split into train and test and that they are in their respective folders (i.e., n number of folders, which corresponds to the number of classes).

## Callbacks
Using the OvA approach, multiple binary models are trained in parallel. For this reason, it is difficult to use TensorFlow's EarlyStopping class while monitoring "val_acc" as there is no "val_acc". Instead, you have "val_model1_acc", "val_model2_acc", etc. 

To solve this, this project developed a custom callback class to monitor the average 'val_acc' across the models. This class can be used with minimal or no
modification to suit your use case.

## How to use
To use this architecture on a multiclass problem, follow the steps below:

1. Clone the repository using `git clone <url>`
2. Navigate to the directory using `cd Multilabel_classification`
3. Modify the `train_path` in the `config.yaml` file to your own training directory.
4. Modify the `val_path` in the `config.yaml` file to your validation directory.
5. Run the main script using `python main.py`
