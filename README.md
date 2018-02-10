# unsupervised_learning
Implementation of K-means 

# PROJECT DESCRIPTION
1) The following project is a part of Innovaccer.Inc.
2) Given a dataset of various people we need to train model in order to deduplicate the dataset.
3) The dataset's shape is (103,4).

# REQUIREMENTS
1) pandas
2) scikit-learn
3) codecs
4) numpy

# HOW IT WORKS
1) Download the training set.
2) Change the path in line 26 and 27 of dedupe.py to the directory where the downloaded training set is located.
3) The final output is the number of unique entries in the dataset that the model predicts by providing unique cluster numbers to
   each entry. The duplicates are having same cluster numbers.
4) The clustering is exported to a final.csv file which has all the attributes of training dataset along with the predicted cluster
   numbers appended. (change the path while exporting to csv in line 84)
