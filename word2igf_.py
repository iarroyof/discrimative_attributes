from __future__ import print_function
import multiprocessing
from functools import partial

from sklearn.feature_extraction.text import CountVectorizer
import argparse
import logging
from time import time
import numpy as np
import codecs
from gensim import corpora, matutils
from gensim.models import TfidfModel, LsiModel
import os
import ntpath
from pathlib import Path
from sys import stderr
from scipy.sparse import csc_matrix
from scipy.sparse import hstack
from six import iteritems
import shutil
from joblib import Parallel, delayed

from pdb import set_trace as st
# Display progress logs on stdout
logging.basicConfig(level = logging.INFO,
                    format = '%(asctime)s %(levelname)s %(message)s')

mark = "%%%_"


def rm_words(user_input, stop_words, out_file):
    """Sanitize using intersection and list.remove()"""
    # Downsides:
    #   - Looping over list while removing from it?
    #     http://stackoverflow.com/questions/1207406/remove-items-from-a-list-while-iterating-in-python

    stop_words = set(map(str.lower, stop_words))
    with codecs.open(out_file, mode = "a", encoding = 'latin-1', errors = 'replace') as f:
        for sw in stop_words.intersection(user_input):
            while sw in user_input:
                user_input.remove(sw)

        f.write("%s\n" %  " ".join(user_input))


def du(path):
    """disk usage in human readable format (e.g. '2,1GB')"""
    import subprocess
    return subprocess.check_output(['du','-sh', path]).split()[0].decode('utf-8')


def sublinear(x):                                              
    return np.log2(x) + 1


def binary(x):
    return 1.0


def freq(x):
    return x

class windowStreamer(object):
    def __init__(self, dictionary, input_file, vectorizer, wsize=10):
        self.file_name = input_file
        self.analyzer = vectorizer.build_analyzer()
        self.tokenizer = vectorizer.build_tokenizer()
        self.wsize = wsize

    def __iter__(self):
        for line in codecs.open(self.file_name, mode = "r", encoding = 'latin-1', errors = 'replace'):
            ln = self.tokenizer(line.lower())
            try:
                for i, _ in enumerate(ln):
                    try:
                        #word = ln[i + self.wsize]
                        word = mark + ln[i]
                    except KeyError:
                        continue

                    #s=" ".join(ln[i:i + self.wsize] + ln[i + self.wsize + 1:i + self.wsize*2 + 1])
                    w = ln[i - self.wsize:i] + ln[i + 1:i + (self.wsize + 1)]
                    s = " ".join(w)
                    wi = [word] + self.tokenizer(" ".join(self.analyzer(s)))
                    bow = dictionary.doc2bow(wi)
                    if len(wi) < 2:
                        #stderr.write("%s\n" % wi)
                        continue

                    yield bow
                    
            except IndexError:
                break
                             

class streamer(object):
    def __init__(self, file_name, vectorizer = None, only_tokens=False):
        self.file_name=file_name
        self.analyzer=vectorizer.build_analyzer()
        self.tokenizer=vectorizer.build_tokenizer()
        self.only_tokens=only_tokens

    def __iter__(self):
        if self.only_tokens:
            for s in open(self.file_name, mode = 'r', encoding = 'latin-1', errors = 'replace'):
                yield self.tokenizer(s.lower())
        else:
            for s in open(self.file_name, mode = 'r', encoding = 'latin-1', errors = 'replace'):
                yield self.tokenizer(" ".join(self.analyzer(s.lower()))) + [mark + w for w in self.tokenizer(s.lower())]


def save_sample(win, dictionary, out_dataset="word_dataset", fsubsamp=50, op="sum", verbose=False):
    try:
        word=[term for term in [dictionary[idx] for idx, weight in win] if mark in term][0].strip(mark)
        context = win
    except IndexError:
        return None
    
    if not word.isalpha():
        return None

    sshape=(max(dictionary.keys()) + 1, 1)
    path = out_dataset + "/" + word
  
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            if verbose: stderr.write("Sample couldn't be stored: %s\n" % word)
            return None
      
        rows=np.array([i for i, f in context])
        data=np.array([f for i, f in context])

        lock.acquire()       

        D=np.memmap(path + "/vector", dtype='float32', mode='w+', shape=data.shape)
        R=np.memmap(path + "/rows", dtype='int32', mode='w+', shape=data.shape)
        K=np.memmap(path + "/n_samples", dtype='int32', mode='w+', shape=(1,))
        Cs=np.memmap(path + "/c_shape", dtype='int32', mode='w+', shape=(1,))
        D[:] = data[:]
        R[:] = rows[:]
        Cs[:] = rows.shape[0]
        K+=1 # Asigning directly '1' losses the memmap variable
        del Cs, D, R, K

        lock.release()

        return None
    else:
        lock.acquire()
        K=np.memmap(path + "/n_samples", dtype='int32', mode='r+', shape=(1,))
    
    # limit the number of samples stored for each word to 'fmax'
        if K[0] >= fsubsamp:
            del K
            lock.release()
            return None

        else:
        
            rows=np.array([i for i, f in context])
            data=np.array([f for i, f in context])
            cols=np.array([0]*int(data.shape[0]))

            sample=csc_matrix((data, (rows, cols)), shape=sshape)

            Cs = np.memmap(path + "/c_shape", dtype='int32', mode='r+', shape=(1,))
            cshape=(Cs[0], )
            D=np.memmap(path + "/vector", dtype='float32', mode='r', shape=cshape)
            R=np.memmap(path + "/rows", dtype='int32', mode='r', shape=cshape)
            C=np.array([0]*int(R.shape[0]))

            centroid=csc_matrix((D, (R, C)), shape=sshape)
            del D, R

            if op == "ol": # On-line mean
                context = centroid + (sample - centroid)/(K[0] + 1.0)

            if op == "avg":
                context = (sample + centroid)/(K[0] + 1.0)

            if op == "sum":
                context = sample + centroid

            data=context.data
            rows=context.indices
            cshape=(rows.shape[0], )
            D=np.memmap(path + "/vector", dtype='float32', mode='w+', shape=cshape)
            R=np.memmap(path + "/rows", dtype='int32', mode='w+', shape=cshape)
            D[:] = data[:]
            R[:] = rows[:]
            Cs[:] = context.data.shape[0]
            K+=1 
        
            del Cs, K, D, R
            lock.release()

class stream_vectors(object):
    def __init__(self, path="word_dataset"):
        self.path=path

    def __iter__(self):
        for word_dir, _, vector_files in os.walk(self.path):
            if vector_files==[]:
                continue

            Cs = np.memmap(word_dir + "/c_shape", dtype='int32', mode='r', shape=(1,))
            cshape=(Cs[0], )
            D=np.memmap(word_dir + "/vector", dtype='float32', mode='r', shape=cshape)
            R=np.memmap(word_dir + "/rows", dtype='int32', mode='r', shape=cshape)
            
            yield [(r, d) for r, d in zip(R, D)]

    

def wind2lsa(doc, dim):
    v=np.zeros((dim,))
    try:
        for index, value in doc:
            v[index] = value
    except:
        return v    
    return v


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Computes Cross-Entropy (TFIDF) weights of a raw text dataset and stores the model.')
    parser.add_argument("--dataset", help = "The path to the raw text dataset file", required = True)
    parser.add_argument("--cout", help = "The path to the cross-entropy output model file (default='output.vec')",
                                                                                            default = "output.vec")
    parser.add_argument("--tmp", help = "The path to the temporary files (default='./tmp')",
                                                                                            default = "./tmp")
    parser.add_argument("--fmin", help = "The minimum word frequency considered to embed (default = 3).",
                                                                                            default = 3, type = int)
    parser.add_argument("--fmax", help = "The maximum word frequency portion considered to embed between [0.0, 1.0] (default = 0; no limit).",
                                                                                            default = 0, type = float)
    parser.add_argument("--fsubsamp", help = "The maximum size of context samples for each word (default=50).", 
                                                                                            default = 50, type = int)
    parser.add_argument("--wsize", help = "The size of the sliding window (default=10).", default = 10, type = int)
    parser.add_argument("--tf", help = "TF normalization: frequency, binary, sublinear (default='frequency').", 
                                                                                            default = "frequency")
    parser.add_argument("--combiner", help = "Combination operation among contexts of a word {'sum':summation, 'avg': mean, 'ol': online_mean} (default='sum').",
                                                                                            default = "sum")
    parser.add_argument("--stop", help = "Toggles stop words stripping.", action = "store_true")
    parser.add_argument("--char", help = "Toggles character n-grams instead of word n-grams (the default).", 
                                                                                            action = "store_true")
    parser.add_argument("--lsa", help = "Output embeddings dimension (default = 300).", default = 300, type = int)
    parser.add_argument("--n_gramI", help = "Inferiror n-gram TF--IDF computation (default = 2).", default = 2, type = int)
    parser.add_argument("--n_gramS", help = "Superiror n-gram TF--IDF computation (default = 6).", default = 6, type = int)
    parser.add_argument("--replace", help = "Toggles replace stored temporal files with new ones.", action = "store_true")
    parser.add_argument("--keep_tmp", help = "Keep temporal files by ending the embedding.", action = "store_true")
    args = parser.parse_args()

    print("Creating LSA sliding windows by using following params:\n%s\n" % vars(args))
    # Functions for computing TF
    wlocal={"frequency": freq, "binary": binary, "sublinear": sublinear}

    TEMP_FOLDER=args.tmp
    if not os.path.isdir(TEMP_FOLDER):
        os.makedirs(TEMP_FOLDER)

    tmp_name = TEMP_FOLDER + '/' + ntpath.basename(args.dataset) + '_fmin-' + str(args.fmin) + '_fmax-' + str(args.fmax) + '_tf-' + args.tf + '_stop' + str(args.stop) + \
                                  '_char-' + str(args.char) + '_dim-' + str(args.lsa) + '_wsize-' + str(args.wsize) + '_n-' + str(args.n_gramI) + '_N-' + str(args.n_gramS) + '_combiner-' + args.combiner
    if not os.path.isdir(os.path.dirname(args.cout)):
        stderr.write("\nOutput directory does not exist...")
        exit()
    input_ = ""
    t0 = time()
    ta = time()    

    vectorizer = CountVectorizer(analyzer = 'char', ngram_range = (args.n_gramI, args.n_gramS), 
                                            strip_accents = 'unicode')

    if not os.path.isdir(TEMP_FOLDER + "/word_dataset") and args.stop and \
                                            (args.replace or not Path(TEMP_FOLDER + "/file_filtered.nstop").is_file()):
        stderr.write("\nFiltering stop words from input file..")
        with open("stop_words.txt", mode = 'r', encoding = 'latin-1', errors = 'replace') as f:
            stopwors = f.read().strip().split('\n')

        stream=streamer(args.dataset, vectorizer = vectorizer, only_tokens=True)
        Parallel(n_jobs=-1)(delayed(rm_words)(line, stopwors, TEMP_FOLDER + "/file_filtered.nstop") for line in stream)
        #os.system(stopw_command)

        if Path(TEMP_FOLDER + "/file_filtered.nstop").is_file():
            input_ = TEMP_FOLDER + "/file_filtered.nstop"
            stderr.write("\nInput file filtered from stop words in %f min %f seg\n" % ((time() - t0)/60.0, time() - t0))

        else:
            stderr.write("\nNo filtered file could be created... aborting\n")
            exit()
    elif not os.path.isdir(TEMP_FOLDER + "/word_dataset"):
        input_ = args.dataset
    #else:
    #    input_ = args.dataset

    t0 = time()
        # Create vectorizer for shattering text into n-gram characters.
    
    my_file = Path(tmp_name + '.dict')
    if (not os.path.isdir(TEMP_FOLDER + "/word_dataset") and not my_file.is_file()) or args.replace:
        	# This streamer returns a dictionary over the raw input file. The 
        	# 'vectorizer' provides analyzer and tokenizer from which depends
        	# the resulting dictionary, e.g. tokenizing with character n-grams.
        corpus = streamer(input_, vectorizer = vectorizer)
        dictionary = corpora.Dictionary(corpus)

        if not Path(TEMP_FOLDER + "/file_filtered.bad").is_file() or args.replace:
            rare_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) 
                                                        if docfreq <= args.fmin 
                                                        and dictionary[tokenid].startswith(mark)] if args.fmin > 0 else []

            max_f = max([f for tokenid, f in iteritems(dictionary.dfs) if dictionary[tokenid].startswith(mark)])
            freq_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) 
                                                        if docfreq >= args.fmax * max_f 
                                                        and dictionary[tokenid].startswith(mark)] if args.fmax > 0 else []
            if rare_ids + freq_ids != []:
                bad_types = [dictionary[idx].strip(mark) for idx in rare_ids + freq_ids]
            else:
                bad_types = []

            #with codecs.open(TEMP_FOLDER + "/badtypes", mode = "w", encoding = 'latin-1', errors = 'replace') as f:
            #    for t in bad_types:
            #        f.write("%s\n" % t.strip(mark))

            t0 = time()
            stderr.write("\nRemoving frequent/rare words from input file...")
            
            #os.system(badt_command) # Remove frequent and rare words
            if bad_types != []:
                stream=streamer(input_ , vectorizer = vectorizer, only_tokens=True)
                Parallel(n_jobs=-1, verbose=1, backend="threading")(delayed(rm_words)(line, bad_types, 
                                                                    TEMP_FOLDER + "/file_filtered.bad") for line in stream)
                if not os.stat(TEMP_FOLDER + "/file_filtered.bad").st_size == 0:
                    input_ = TEMP_FOLDER + "/file_filtered.bad"
                    dictionary.filter_tokens(rare_ids + freq_ids)
                    dictionary.compactify()
                    stderr.write("\nInput file filtered from frequent/rare words in %f min %f seg\n" % ((time() - t0)/60.0, time() - t0))

            elif not os.path.isdir(TEMP_FOLDER + "/word_dataset"):
                input_ = args.dataset

        dictionary.save(tmp_name + '.dict')
        stderr.write("\nDictionary created in %f min %f seg\n" % ((time() - t0)/60.0, time() - t0))

    dictionary = corpora.Dictionary.load(tmp_name + '.dict')
    t0 = time()

    #stderr.write("\nSerializing sparse corpus\n")
    my_file = Path(tmp_name + '.mm')

    if (not os.path.isdir(TEMP_FOLDER + "/word_dataset") and not my_file.is_file()) or args.replace:
            # This stramer returns generator of sliding windows already vectorized 
            # with word counts 
        sdata = windowStreamer(dictionary = dictionary, input_file = input_, vectorizer = vectorizer, 
                                                                                        wsize = args.wsize)
        ####corpora.MmCorpus.serialize(tmp_name + '.mm', sdata)
        stderr.write("\nSparse BoW corpus serialized in %f min %f seg\n" % ((time() - t0)/60.0, time() - t0))
        
    ####corps = corpora.MmCorpus(tmp_name + '.mm')
        corps = sdata
    #st()    
    if not os.path.isdir(TEMP_FOLDER + "/word_dataset"):
        try:
            os.makedirs(TEMP_FOLDER + "/word_dataset")
        except OSError:
            if verbose: stderr.write("Sample couldn't be stored: %s\n" % word)
            exit()

        t0 = time()
        stderr.write("\nFitting entropy model for sparse word embeddings (TF-IDF)\n")
        tfidf = TfidfModel(corps, normalize = True, wlocal = wlocal[args.tf])
        tfidf_corpus = tfidf[corps]
        stderr.write("\nSparse word embeddings created in %f min %f seg\n" % ((time() - t0)/60.0, time() - t0))

        t0=time()
        stderr.write("\nMerging sparse entropy word embeddings (TF-IDF)\n")

        def init(l):# Create processing global lock
            global lock
            lock = l
        # Initialize constant arguments of 'save_sample'...
        save=partial(save_sample, out_dataset = TEMP_FOLDER + "/word_dataset", dictionary=dictionary, 
                                               fsubsamp = args.fsubsamp, op=args.combiner, verbose=False)
        l = multiprocessing.Lock()
        #pool = multiprocessing.Pool(initializer=init, initargs=(l,),processes=20)
        pool = multiprocessing.pool.ThreadPool(initializer=init, initargs=(l,), processes=20)
        pool.imap(save, tfidf_corpus, chunksize=10)
        pool.close()
        pool.join()

        #for win in tfidf_corpus:
        # The first item of a window in the gensim corpus contains the lexical type associated to it.
        #   save_sample(win=win, out_dataset = TEMP_FOLDER + "/word_dataset", 
        #                        dictionary=dictionary, fsubsamp = args.fsubsamp, op=args.combiner, verbose=False)

    # Once entropy vectors have been combined, let's stream them to a new corpus
    stderr.write("\nSparse word embeddings memmaped in %f min %f seg\n" % ((time() - t0)/60.0, time() - t0))
    t0 = time()

    #tfidf_vectors=stream_vectors(path = TEMP_FOLDER + "/word_dataset")
    tfidf_corpus=stream_vectors(path = TEMP_FOLDER + "/word_dataset")
    #corpora.MmCorpus.serialize(tmp_name + '_entropy.mm', tfidf_vectors)

    #stderr.write("\nSparse entropy matrix serialized in %f min %f seg\n" % ((time() - t0)/60.0, time() - t0))

    #tfidf_corpus = corpora.MmCorpus(tmp_name + '_entropy.mm')
    stderr.write("\nEntropy model fitted in %f min %f seg\n" % ((time() - t0)/60.0, time() - t0))
    t0 = time()
    stderr.write("\nFitting latent orthogonal basis...\n")
    lsi = LsiModel(tfidf_corpus, id2word = dictionary, num_topics = args.lsa)
    corpus_lsi = lsi[tfidf_corpus]

    print("Words embedded into orthogonal basis in %f min %f seg\n" % ((time() - t0)/60.0, time() - t0))

    t0 = time()
    print ("Saving vectors ... \n")

    with codecs.open(args.cout, mode = "w", encoding = 'latin-1', errors = 'replace') as f:
        f.write("%s %s\n" % (lsi.docs_processed, lsi.num_topics))
        for v, context in zip(corpus_lsi, tfidf_corpus):
            word=[term for term in [dictionary[idx] for idx, weight in context] if mark in term][0].strip(mark)
            f.write("%s %s\n" % (word, np.array2string(wind2lsa(v, lsi.num_topics), 
                                formatter={'float_kind':lambda x: "%.6f" % x}, max_line_width=20000).strip(']').strip('[') ))

    if not args.keep_tmp:
        S=du(TEMP_FOLDER)
        shutil.rmtree(TEMP_FOLDER)
        print("Temporal files removed: size %s ...\n" % S)

    print("Word embeddings saved at %s ...\nTotal time: %.4f min %.4f seg\n" % (args.cout, (time() - ta)/60, time() - ta))
