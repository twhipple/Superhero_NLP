# Superhero_NLP

## "Work in Progress"


![picture1](https://raw.githubusercontent.com/twhipple/Superhero_NLP/main/Images/yulia-matvienko-kgz9vsP5JCU-unsplash.jpg)


*Superheros come to save the day! Source: Marjan Blan, unsplash.com*


## Intro
Trying to determine if a superhero is good or bad from text.
A Natural Language Processing Model based on a dataset from Kaggle Kernels.


## README Outline
* Introduction 
* README Outline
* Repo Contents
* Libraries and Prerequisites
* Feature and Definitions
* Models
* Conclusions
* Future Work
* Built With, Contributors, Authors, Acknowledgments


![picture2](https://raw.githubusercontent.com/twhipple/Superhero_NLP/main/Images/marjanblan-_CjdyBNNVio-unsplash.jpg)

*Using 'text' not 'photographs' to determine who is who in the hero world! Source: Marjan Blan, Unsplash.com*


## Repo Contents
This repo contains the following:
* README.md - this is where you are now!
* Disaster_Tweets_Notebook.ipynb - the Jupyter Notebook containing the finalized code for this project.
* Disaster_Tweets_Notebook_Again.ipynb - A notebook I used to play around, explore, and learn.
* LICENSE.md - the required license information.
* sample_submission.csv
* test.csv
* train.csv
* CONTRIBUTING.md 
* Images - contains the fun images to this repo.
* my_submission.csv - my results after running the test through my model.



## Libraries & Prerequisites
These are the libraries that I used in this project.

* import pandas as pd
* import numpy as np
* import matplotlib.pyplot as plt
* %matplotlib inline
* import seaborn as sns

* import string, re
* import nltk
* from nltk import FreqDist, word_tokenize
* from nltk.corpus import stopwords 

* from sklearn.feature_extraction.text import TfidfVectorizer
* from sklearn.model_selection import train_test_split
* from sklearn.metrics import accuracy_score
* from sklearn.ensemble import RandomForestClassifier
* from sklearn.naive_bayes import MultinomialNB



## Features
These are the features of this dataset.

Columns
* id - a unique identifier for each tweet
* text - the text of the tweet
* location - the location the tweet was sent from (may be blank)
* keyword - a particular keyword from the tweet (may be blank)
* target - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)


## Models
Naive Bayes and Random Forest



## Conclusions
I first tried to separate all the words in the text file, then get rid of numbers and symbols. Then I spent some time trying to get rid of the https links and all of the webpages. Finally I made all the words lowercase. I tried two different Tweet processing methods but neither of them worked - perhaps because the string of text had been changed to a list.

Using the TfidfVectorizer along with my nltk cleaning that was somehow joined again to a string I was able to improve my testing accuracy for the Naive Bayes which went from 79.94% to 80.51%. The Random Forest testing accuracy also went up from 77.73% to 78.83%. Neither are great and both are only slight improvements, but nevertheless heading in the right direction. I still have some more cleaning issues to deal with - since a bunch of the word vectors seemed to be just letters.


## Future Work
There is so much more I want to do! I still have a lot to learn about text pre-processing - including how to put it all into a nice easy function. I still need to clean out emoji's and other types of tweet junk.


![Picture3](https://raw.githubusercontent.com/twhipple/Superhero_NLP/main/Images/marjanblan-8I-ht65iRww-unsplash%20(1).jpg)

*Is this one of the good guys! Source: Yulia Matvienko, unsplash.com*


## Built With:
Jupyter Notebook
Python 3.0
scikit.learn


## Contributing
Please read CONTRIBUTING.md for details


## Authors
Thomas Whipple


## License
Please read LICENSE.md for details


## Acknowledgments
Kaggle - This dataset was created by the company figure-eight and originally shared on their ‘Data For Everyone’ website here.
