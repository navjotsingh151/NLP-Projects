# Library Import

import pandas as pd
import numpy as np
from matplotlib import pyplot
import seaborn as sns
from nltk.corpus import stopwords
import string
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


##################### Training The model ##################

train = pd.read_csv(r"D:\Github\NLP-Projects\Kaggle Word2Vec\Files\labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting = 3)
print(train.head(10))

## Cleaning and refining each review
stop= stopwords.words("english")
punct=list(string.punctuation)

def clean_text(text):
    
    #Removing Tags
    soup = BeautifulSoup(text,"lxml")
    text = soup.get_text()
    
    #lower the text for simplicity
    text=text.lower()
    
    #removing Punctuation and stop words
    list_word=""
    for word in text.split():
        if word not in stop and word.isalpha():
            word = [wrd for wrd in word if wrd not in punct and wrd.isalpha()]
            word = ''.join(word)
            list_word=list_word + word +" "
    return list_word

#Building Clean Review
train['clean_review']= train['review'].apply(lambda x :clean_text(x))
clean_review=[]
for review in train['clean_review']:
    clean_review.append(review)

print ("Creating the bag of words...\n")

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_review)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

print (train_data_features.shape)


# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print (vocab)

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print (count, tag)

print ("Training the random forest...")

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["sentiment"] )

##################  Testing the model and creating submission file ##################

# Read the test data
test = pd.read_csv(r"D:\Github\NLP-Projects\Kaggle Word2Vec\Files\testData.tsv", header=0, delimiter="\t", \
                   quoting=3 )

# Verify that there are 25,000 rows and 2 columns
print (test.shape)

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])

print ("Cleaning and parsing the test set movie reviews...\n")
test['clean_review']=test['review'].apply(lambda x : clean_text(x))
clean_test_reviews = [] 

for rev in test['clean_review']:
    clean_test_reviews.append(rev)

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )

################## EOF ####################
