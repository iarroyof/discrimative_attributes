
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import logging
import os
import numpy as np
from scipy import io
from scipy.sparse import bsr_matrix
import codecs
import sh
from pdb import set_trace as st

class streamer(object):
    def __init__(self, file_name):
        self.file_name=file_name
    def __iter__(self):
        for s in open(self.file_name):
            try:
                yield s.strip().split("\t")[1]
            except:
                continue


def save_dense(filename, v):
    # cout + "/" + w0, c
    word=filename.split("/")[-1]
    cout=filename.replace("/" + word, '')  

    with codecs.open(cout, mode = "a+", encoding = 'latin-1', errors = 'replace') as f:
        f.write("%s %s\n" % (word, np.array2string(v,
                                formatter={'float_kind':lambda x: "%.6f" % x}, max_line_width=20000).strip(']').strip('[') ))

def load_sparse_bsr(filename):
    loader = np.load(filename)                                                               
    return bsr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def save_sparse_bsr(filename, array):
    # note that .npz extension is added automatically
    array=array.tobsr()
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


class vector_space(object):
    def __init__(self, directory):
        directory=os.path.normpath(directory) + '/'
        self.words = {word.replace(".npz", ''): directory + word for word in os.listdir(directory)}

    def __getitem__(self, item):
        return load_sparse_bsr(self.words[item])


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


def fit_vectors(dataset, cout, tf="binary", stop=True, analyzer="word", lsa=0, uniq=False):
    """ The input 'dataset' must be a tab separated dictionary file with 
        following format:

        <term_0>\t<definition_0>
        <term_0>\t<definition_1>
        ...
        <term_0>\t<definition_m>
        ...
        <term_n>\t<definition_0>
        <term_n>\t<definition_1>
        ...
        <term_n>\t<definition_m>

        'cout' must be a directory where the output tfidf embeddings will be 
        stored
        """

    types = []
    corpus=streamer(dataset)

    from sklearn.pipeline import Pipeline

    vectorizer = TfidfVectorizer(min_df=1,
                                 encoding="utf-8",
                                 decode_error="replace",
                                 lowercase=True,
                                 analyzer=analyzer,
                                 binary= True if tf.startswith("bin") else False,
                                 sublinear_tf= True if tf.startswith("subl") else False,
                                 stop_words= "english" if stop else None)
    if lsa != 0:
       from sklearn.decomposition import TruncatedSVD
       svd_model = TruncatedSVD(n_components=lsa, 
                         algorithm='randomized',
                         n_iter=10, random_state=42)

       vectorizer = Pipeline([('tfidf', vectorizer),
                                      ('svd', svd_model)])
       save=save_dense
    else:
       save=save_sparse_bsr

    tfidf = vectorizer.fit(corpus)
#    if lsa == 0:
    transform = tfidf.transform
#    else:
#        transform = tfidf.transform

    w0=""

    if not os.path.exists(cout) and lsa == 0:
        os.makedirs(cout)    

    if not uniq:
        with open(dataset, "r") as f:
            N = 0
            for ln in f:
                try:
                    w, d = ln.strip().split("\t")
                
                except:
                    print("Problem with input line: %s" % ln)
                    continue

                if w == w0 and w not in types:
                    v = transform([d])
                    c = c + (v - c)/(N + 1)
                    w0 = w
                    N+=1

                elif w0 != "" and w != w0 and w not in types:
                    types.append(w)
                    N = 0.0
                    save(cout + "/" + w0, c)
                    w0 = w
                    c = transform([d])

                elif w0 == "" and w not in types:
                    c = transform([d])
                    w0 = w
                    N = 1.0
    else:
        with open(dataset, "r") as f:
            if lsa > 0:
                #word=(cout + "/" + w).split("/")[-1]
                #out_file=(cout + "/" + w).replace("/" + word, '')
                with open(cout, "w+") as fo:
                    fo.write(" \n")
            i = 0

            for ln in f:
                try:
                    w, d = ln.strip().split("\t")
                    save(cout + "/" + w, transform([d]))
                    i += 1
                except:
                    print("Problem with input line: %s" % ln.replace("\t", "^T^"))
                    continue
            if lsa > 0:
                header = str(i) + " " + str(lsa)
                sh.sed("-i", "1s/.*/" + header + "/", cout)
                
    return True
