# orgnanic_compounds-
Classifying organic compounds 

Using data from the University of British Columbia we built a neural network for predicting the class of organic compounds ranging from fatty acyls to tannins. There are 219 unique organic compounds in the data making it difficult for classification.  

We first split the organic compound data into 65% training and 35% reserved for testing. The predictors used in the neural network were variants of each compounds’ partition coefficient, which is a measure of the difference in the solubility of a compound, as well as each compounds’ pKa.  For reference, the pKa measures the acid strength in a chemical solution. The response (y output) was the class of each compound.  

For classification, we created a keras multi-perceptron model with two hidden layers and a stochastic gradient descent optimizer to minimize loss. The model was only able to classify 12.2% of the classes correctly but probably expected due to the high number of classes.



