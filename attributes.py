
import numpy as np
from sklearn.metrics import f1_score
from joblib import Parallel, delayed
import tempfile
import os, codecs
from sys import stderr
import logging, argparse
import itertools
from scipy.sparse.linalg import norm as spnorm
from mutual_inf import calc_MI as MI


def pmax(a, b):
    return np.max([a, b], axis=0)

def pmin(a, b):
    return np.min([a, b], axis=0)


def sparse_dot(a, b):
    return a.dot(b.transpose()).toarray()[0][0]


def sparse_max(A, B):
    #  It is the standard semantics for disjunction (and) in Gödel fuzzy logic
    BisBigger = B > A
    return A - A.multiply(BisBigger) + B.multiply(BisBigger)


def sparse_min(A, B):
    #  It is the standard semantics for conjunction (or) in Godel fuzzy logic
    BisBigger = B < A
    return A - A.multiply(BisBigger) + B.multiply(BisBigger)


class streamer(object):                     
    def __init__(self, file_name, model, normalize=False, sparse=False):
        self.file_name=file_name
        self.model=model
        self.normalize=normalize
        if not sparse:
            self.norm = np.linalg.norm

        else:
            self.norm = spnorm
     
    def __iter__(self):
        if not self.normalize:
            for i, ln in enumerate(codecs.open(self.file_name, mode = 'r', 
                                    encoding = 'latin-1', errors = 'replace')):
                try:
                    wa, wb, wq, s = tuple(ln.strip().split(','))
                except ValueError:
                    wa, wb, wq = tuple(ln.strip().split(','))
                    s = '1'

                if not isinstance(self.model, tuple):
                    try:
                        yield self.model[wa], self.model[wb], self.model[wq], s, i
                    except KeyError as e:
                        yield None, "%s" % e, None, s, i
                else:
                    try:
                        yield self.model[0][wa], self.model[0][wb], self.model[0][wq], s, i
                    except KeyError as e:
                        pass

                    try:
                        yield self.model[1][wa], self.model[1][wb], self.model[1][wq], s, i
                    except KeyError as e:
                        yield None, "%s" % e, None, s, i

        else:
            for i, ln in enumerate(codecs.open(self.file_name, mode = 'r', 
                                    encoding = 'latin-1', errors = 'replace')):
                try:
                    wa, wb, wq, s = tuple(ln.strip().split(','))
                except ValueError:
                    wa, wb, wq = tuple(ln.strip().split(','))
                    s = '1'

                if not isinstance(self.model, tuple):
                    try:
                        a=self.model[wa]
                        b=self.model[wb]
                        q=self.model[wq]
                        yield a/self.norm(a), b/self.norm(b), q/self.norm(q), s, i

                    except KeyError as e:
                        yield None, "%s" % e, None, s, i
                else: # Change by a for loop over the tuple of models for more than two models
                    try:
                        a=self.model[0][wa]
                        b=self.model[0][wb]
                        q=self.model[0][wq]
                        yield a/self.norm(a), b/self.norm(b), q/self.norm(q), s, i

                    except KeyError as e:
                        try:
                            a=self.model[1][wa]
                            b=self.model[1][wb]
                            q=self.model[1][wq]
                            yield a/self.norm(a), b/self.norm(b), q/self.norm(q), s, i

                        except KeyError as e:
                            yield None, "%s" % e, None, s, i

def unsup_attributes(wa, wb, wq, s, i, method="cone", normalize=False, bin=True, th = 0.7, beta=0.5, sparse=False):
    cone = 0.5
    if wa is None:
        return i, None, s

    if not sparse:
        snorm = np.linalg.norm
        dotp = np.dot
   
    else:
        snorm = spnorm
        dotp = sparse_dot

    if method == "cone":
        alpha = snorm(wq - wb)/snorm(wa - wb)
        delta = -2*abs(alpha - cone) + 1.0
        if bin:
            # −2|λ − 0.5| + 1
            if delta < th: 
                return i, 0.0, s
            else:
                return i, 1.0, s
        else:
            return i, delta, s
 
    elif method == "mean":
        mv = (wa + wb) * 0.5 # Compute the mean vector
        if normalize:
            co = dotp(wq, mv)
        else:
            co = dotp(wq, mv)/(snorm(wq)* snorm(mv))

        if bin:
            if co < th:
                return i, 0.0, s
            else:
                return i, 1.0, s
        else:
            return i, co, s


    elif method == "sum":
        sv = wa + wb # Compute the mean vector
        if normalize:
            co = dotp(wq,sv)
        else:
            co = dotp(wq,sv)/(snorm(wq)* snorm(sv))

        if bin:
            if co < th:
                return i, 0.0, s
            else:
                return i, 1.0, s
        else:
            return i, co, s

    elif method == "fuzzy":
        if sparse:
            smax = sparse_max
            smin = sparse_min
        else:
            smax = pmax
            smin = pmin
        #              union                    intersection
        fuzzy = beta * smin(wa, wb) + (1 - beta)* smax(wa, wb)
        #intersection = np.array([(a*b)/(a + b - a*b) for a, b in zip(wa, wb) if a != 0 and b != 0])
        delta = dotp(wq,fuzzy)/(snorm(wq)* snorm(fuzzy))
        #delta = snorm(smin(wq, fuzzy))
        if bin:
            if  delta < th:# co < th*snorm(wq):
                return i, 0.0, s
            else:
                return i, 1.0, s
        else:
            return i, delta, s

    elif method == "triang":
        # cos−1(x⋅z)≤cos−1(x⋅y)+cos−1(y⋅z)
        ab = dotp(wa,wb)
        ab = 1.0 if ab > 1.0 else np.arccos(ab) # Compute de difference arc a-b
        aq = dotp(wa,wq)
        aq = 1.0 if aq > 1.0 else np.arccos(aq) # Compute de difference arc a-q left hand side of the triangle inequality
        bq = dotp(wb,wq)
        bq = 1.0 if bq > 1.0 else np.arccos(bq) # Compute de difference arc b-q
        arc = ab + bq              # right hand side of the triangle inequality 
        #co = dotp(wq,rv)/(snorm(wq)* snorm(rv))
        if bin:
            if arc/np.pi > th:
                # It means that wa and wb are almost (th) antiparallel so they cannot 
                # produce a linear combination of them cancelling all possible attributes.
                return i, 0.0, s

            else:
                if (aq - arc)/np.pi > th:
                    return i, 0.0, s

                else:
                    return i, 1.0, s
        else:
            return i, aq - arc, s
            

    elif method == "arcone":
        # cos−1(x⋅z)≤cos−1(x⋅y)+cos−1(y⋅z)
        ab = dotp(wa,wb)
        ab = 1.0 if ab > 1.0 else np.arccos(ab) # Compute de difference arc a-b
        bq = dotp(wb,wq)
        bq = 1.0 if bq > 1.0 else np.arccos(bq) # Compute de difference arc b-q left hand side of the triangle inequality
        
        alpha = bq/ab
        delta = -2 * abs(alpha - cone) + 1.0

        if bin:
        # It means that wa and wb are almost (th) antiparallel so they cannot 
        # produce a linear combination of them cancelling all possible attributes.
        #    return i, 0.0, s
            if delta < th:
                return i, 0.0, s

            else:
                return i, 1.0, s
        else:
            return i, delta, s


    elif method == "infocone":
        ab = MI(wa, wb, sparse=sparse) # The mutual information as a metric
        qb = MI(wq, wb, sparse=sparse)
        alpha = qb/ab
        delta = -2 * abs(alpha - cone) + 1.0

        if bin:
            if delta < th:
                return i, 0.0, s

            else:
                return i, 1.0, s
        else:
            return i, delta, s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Computes attribute sharing degree between word vectors.')
    parser.add_argument("--embeddings", help = "The path to the BIN word embeddings file.", required = True)
    parser.add_argument("--out", help = "The path to the output result file (default='output.res').",
                                                                                            default = "output.res")
    parser.add_argument("--th", help = "The correlation threshold for considering it as a positive identification (default=0.7).", 
                                                                                            type = float, default = 0.7)
    parser.add_argument("--queries", help = "The path to the query file.", required = True)
    parser.add_argument("--normalize", help = "Toggle whther to normalize input vectors.", action = "store_true")
    parser.add_argument("--eval", help = "Toggle whther to evaluate. Default: to write into file.", action = "store_true")
    parser.add_argument("--grid", help = "Toggle whther to generate a grid of experiments.", action = "store_true")
    parser.add_argument("--method", help = "Method for decision whether sharing attributes or not.", default = "arith")
    parser.add_argument("--format", help = "The format of the input model. 'w2v' (default): binary word2vec format; 'mine': my format, so '--embeddings' is a directory or a compressed direcory (uncompressed directory will be much faster to read).}", default = "w2v")
    parser.add_argument("--bin", help = "Toggle whther to return raw decision function or return binary values.", action = "store_true")
    parser.add_argument("--mix", help = "Toggle whther to mix two embedding methods. Give embedding files separated by comma (with no spaces) to the '--embedding's parameter.", action = "store_true")
    parser.add_argument("--hybrid", help = "The hybrid connective parameter. h=0.0 for complete intersection; h=1.0 for complete union (default=0.5).", 
                                                                                            type = float, default = 0.5)
    parser.add_argument("--jobs", help = "The number of computing jobs. (default=-1).", type = int, default = -1)
    args = parser.parse_args()

    bin_file=args.embeddings #"/almac/ignacio/data/reuters/w2igf_H200_w10.bin"
    q_file=args.queries #"/almac/ignacio/DiscriminAtt/training/train.txt"
    normalize=args.normalize # False
    method = args.method
    out_file = args.out
    bin = args.bin

    logging.basicConfig(filename='embedding.log',level=logging.DEBUG)

    logging.debug("Loading word vectors...")
    if args.format == "w2v" and not args.mix:
        from gensim.models.keyedvectors import KeyedVectors
        word_vectors = KeyedVectors.load_word2vec_format(bin_file, binary=True, encoding='utf-8', unicode_errors='replace')
        sp = False

    elif args.format == "mine" and not args.mix:
        import def2tfidf
        word_vectors = def2tfidf.vector_space(bin_file)
        sp = True

    elif args.mix:
        if args.format == "w2v" and args.format != "mine":
            from gensim.models.keyedvectors import KeyedVectors
            embedding_a, embedding_b = bin_file.split(',')
            word_vectors_a = KeyedVectors.load_word2vec_format(embedding_a, binary=True, encoding='utf-8', unicode_errors='replace')
            word_vectors_b = KeyedVectors.load_word2vec_format(embedding_b, binary=True, encoding='utf-8', unicode_errors='replace')
            word_vectors = tuple([word_vectors_a, word_vectors_b])
            sp = False
        else:
            print("Still not supported...\n")
            exit()


    logging.info("Word vectors loaded...")

    logging.info("Loading queries...")

    queries=streamer(q_file, word_vectors, normalize=normalize, sparse=sp)

    if not args.grid:
        par_dic = {"hybrid": [args.hybrid], "lamda": [args.th], "funct": [method]}
        ds = args.embeddings.strip('/').split('/')[-1] + "_"
    else:
        par_dic = {"hybrid": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0], "lamda": [-1.0, -0.7, -0.4, 0.1, 0.0, 0.1, 0.4, 0.7, 1.0],
                "funct": ["cone", "arcone", "fuzzy", "sum", "mean", "triang", "infocone"]}
        ds = ""

    #par_dic = {"hybrid": [0.0, 0.4], "lamda": [-0.4, 0.1],
    #             "funct": ["cone", "arcone"]}#, "embedding": [e.strip() for e in open("embeddings.txt")]}
    qs=[q.strip().split(",") for q in open(q_file, "r")]

    with Parallel(n_jobs=args.jobs, verbose=10) as parallel:
    #R = Parallel(n_jobs=args.jobs)(
    #    delayed(unsup_attributes)(wa, wb, wq, s, i, method=method, normalize=normalize, bin=bin, th=args.th, beta=args.hybrid)
    #                                                                            for wa, wb, wq, s, i in queries)
        ref = {i : (vals, par_dic[vals]) for i , vals in enumerate(par_dic)}
        jobs = [t for t in itertools.product(*[ref[par][1] for par in ref])]
        idx = {ref[i][0]: i for i in ref}

        for job in jobs:
            
            directory = ds + "%s_l-%s_h-%s" % (job[idx["funct"]], job[idx["lamda"]],str(job[idx["hybrid"]]) if job[idx["funct"]] == "fuzzy" else "None")  
       	    if not os.path.exists(directory):
       	       	os.makedirs(directory)
       	       	os.makedirs(directory + "/res")
       	       	os.makedirs(directory + "/ref")
       	    else:
       	        continue

            R=parallel(
                delayed(unsup_attributes)(wa, wb, wq, s, i, method=job[idx["funct"]], normalize=normalize, sparse=sp, 
                                                            bin=bin, th=job[idx["lamda"]], beta=job[idx["hybrid"]])
                                                                                for wa, wb, wq, s, i in queries)
            if not args.eval:                
                with open(directory + "/res/answer.txt", "w") as fa, open(directory + "/ref/truth.txt", "w") as ft:
                    for i, r, t, in  R:
                        rr = int(r) if r is not None else np.random.choice([0, 1])
                        fa.write("%s,%s,%s,%d\n" % (qs[i][0], qs[i][1], qs[i][2], rr))
                        ft.write("%s,%s,%s,%d\n" % (qs[i][0], qs[i][1], qs[i][2], int(t)))
                
    # (16895, 0.0, '1')
            else:
                y_pred=np.array([r[1] for r in R if r[1] is not None])
                y_true=np.array([int(r[2]) for r in R if r[1] is not None])

                print("%s\t%f" % (job, f1_score(y_true, y_pred)))
    
