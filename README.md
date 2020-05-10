# Customer Classification
Dataset: [Statlog (German Credit Data) Data Set](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))

Final project for Machine learning course for Software and Knowledge Engineering, Kasetsart University

### Prerequisite
- Python >= 3.6
- Cuda

install required libraries
```sh
pip install -r requirements.txt
```
run jupyter notebook
```sh
jupyter notebook
```

-----

## Preprocessing

Given the dataset with this many categorical features I have to convert it from nominal to numerical data
in order for it to be usable by various machine learning methods that don't support categorical data.

The choice being One-hot encoding, Label encoding, Binary encoding.
While one-hot is commonly use, in our case we are working with Tree based algorithm like Decision Tree, 
Random forest, and the other boosting methods which is also tree based will be slow down by the large amount
of new features that come from each categorical levels. I ended up using Label encoding for convenience, 
but the best choice in my opinion might be Binary encoding.

The dataset is splitted in to 0.7 train: 0.3 test

I've done 2 tests. The first one using all features. The second one with reduced features.

At first I want to use PCA for dimensionality reduction, but PCA is unsupervised. I want to know which feature is relevant 
and has effect on the prediction. So I use chi-square method to select 8 best features to train the model instead.

## Report Finding

All of the machine learning model trained using following methods:
- Decision Tree
- Random Forest
- Adaboost
- Gradient boost
- XGBoost
- Support vectore machine
- Feed Forward Neural Network

doesn't have significant difference in accuracy apart from Decision Tree that performs worst with accuracy around 62-70%.
My assumpsion is that because the dataset is really small with only 1000 samples and unbalance number between the 2 output labels.
I have played and tweaked with multiple pararmeter for all above methods but did not achieve much better result.

The interesting thing is the cost for Adaboost is significantly higher than the rest.
Adaboost almost classify all sample as a good customer, while barely classify any as bad one, therefore the high cost.

Xgboost on the other hand, on average perform the with best with top accuracy and lowest cost.

After I experiment with using all features to train models, I then use the reduced features dataset.
The result are a little bit better the cost are reduced for most methods and the accuracy increased a little. 


