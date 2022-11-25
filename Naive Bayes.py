import numpy as np
import json
from sklearn.feature_extraction import text


class NaiveBayes(object):

    def __init__(self):
        # load Documents
        x = open('fedpapers_split.txt').read()
        papers = json.loads(x)

        # split Documents
        papersH = papers[0]  # papers by Hamilton
        papersM = papers[1]  # papers by Madison
        papersD = papers[2]  # disputed papers

        # Number of Documents for H, M and D
        nH = len(papersH)
        nM = len(papersM)
        nD = len(papersD)

        '''To ignore certain common words in English that might skew your model, we add them to the stop words 
        list below. You may want to experiment by choosing your own list of stop words, but be sure to keep 
        'HAMILTON' and 'MADISON' in this list at a minimum, as their names appear in the text of the papers 
        and leaving them in could lead to unpredictable results '''

        stop_words = text.ENGLISH_STOP_WORDS.union({'HAMILTON', 'MADISON'})
        # stop_words = {'HAMILTON','MADISON'}
        # Form bag of words model using words used at least 10 times
        vectorizer = text.CountVectorizer(stop_words=stop_words, min_df=10)
        X = vectorizer.fit_transform(papersH + papersM + papersD).toarray()

        '''To visualize the full list of words remaining after filtering out stop words and words used less 
        than min_df times uncomment the following line'''
        # print(vectorizer.vocabulary_)

        # split word counts into separate matrices
        self.XH, self.XM, self.XD = X[:nH, :], X[nH:nH + nM, :], X[nH + nM:, :]

    def _likelihood_ratio(self, XH, XM):  # [5pts]
        '''
        Args:
            XH: nH x D where nH is the number of documents that we have for Hamilton,
                while D is the number of features (we use the word count as the feature)
            XM: nM x D where nM is the number of documents that we have for Madison,
                while D is the number of features (we use the word count as the feature)
        Return:
            fratio: 1 x D vector of the likelihood ratio of different words (Hamilton/Madison)
        '''
        # Estimate probability of each word in vocabulary being used by Hamilton

        # Estimate probability of each word in vocabulary being used by Madison

        # Compute ratio of these probabilities
        XH[:, :] += 1
        total = np.sum(XH, axis=1).reshape(XH.shape[0], 1)
        summy = XH / total
        LH = np.mean(summy, axis=0)
        # print(LH)
        # print(LH.shape)
        XM[:, :] += 1
        total = np.sum(XM, axis=1).reshape(XM.shape[0], 1)
        summy = XM / total
        LP = np.mean(summy, axis=0)
        fratio = LH / LP
        # print(fratio)
        return fratio

    def _priors_ratio(self, XH, XM):  # [5pts]
        '''
        Args:
            XH: nH x D where nH is the number of documents that we have for Hamilton,
                while D is the number of features (we use the word count as the feature)
            XM: nM x D where nM is the number of documents that we have for Madison,
                while D is the number of features (we use the word count as the feature)
        Return:
            pr: prior ratio of (Hamilton/Madison)
        '''
        # Compute prior probabilities

        nH = np.sum(XH)
        nM = np.sum(XM)
        n = nH + nM
        return (nH / n) / (nM / n)

    def classify_disputed(self, fratio, pratio, XD):  # [5pts]
        '''
        Args:
            fratio: 1 x D vector of ratio of likelihoods of different words
            pratio: 1 x 1 number
            XD: 12 x D bag of words representation of the 12 disputed documents (D = 1307 which are the number of features for each document)
        Return:
             1 x 12 list, each entry is H to indicate Hamilton or M to indicate Madison for the corrsponding document
        '''
        a = []
        XD += 1
        for i in range(12):
            a.append("H" if np.prod(XD[i] * fratio) * pratio > 1 else "M")
        return a