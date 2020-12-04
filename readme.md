In this project, I have compared two models for the task of Natural Language Inference on the SNLI dataset.

**Problem Statement** ​-
Given two sentences hypothesis and premise, the task is to predict whether these pair of
statements are entailmen, contradiction, neutral. SNLI (Stanford Natural Language Inference) dataset is used in this project, which has 60000 train
examples and 10000 test examples.

The first model is a Logistic Regression Model with TF-IDF features and the second model is an LSTM model with Glove vectors as word emeddings. The description of both the models is given in the deep_learing_project_report.pdf

To run the saved model, write python main.py in the terminal.



**Preprocessing-**
The hypothesis and premises are pre-processed separately.
Following pre-processing methods are used-

1. Text to lowercase
2. Remove punctuation
3. Remove extra white-spaces
4. Lemmatize word- converts words to its base or dictionary form
5. Drop Stop words- Drop the words which make lesser or no sense in sentiment
    analysis like and, this, the, etc.
    
EXAMPLE: 

Before preprocessing- ​A woman with a rolling luggage waits on a sidewalk.
After preprocessing- ​woman roll luggage wait sidewalk


**Models** ​-

# 1. Logistic regression using TF-IDF vectors

After preprocessing, data is converted to tf-idf vectors. These tf-idf vectors are then used
train the logistic regression model. Using this model I schieved  43.75% accuracy.
Any changes in the model were not making much change in the training accuracy.


# 2. Deep learning models(RNN and LSTM)

After preprocessing, words in each sentence are replaced by a number, and zero padding is
applied. After that pre-trained Glove model is used to used to get the word embeddings.
**GloVe-**
Pretrained Glove models are available on nlp.stanford.edu. In this project, I used a 50-dimensional pre-trained GloVe model.
**LSTM**
To get the model architecture, different values are done on the hyperparameters are tried
which is mentioned in the deep_learing_project_report.pdf.



