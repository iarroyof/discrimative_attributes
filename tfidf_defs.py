
import logging
import def2tfidf
import argparse
import os

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Computes mutual information (TFIDF) weights of a definitional dataset and stores the model.')
    parser.add_argument("--dataset", help="The path to the definitional text dataset file",
                                                                required=True)
    parser.add_argument("--cout", help="The path to the cross-entropy output model file",
                                                                default="output_")
    parser.add_argument("--minc", help="The minimum word frequency considered to compute CE weight.",
                                                                default=1, type=int)
    parser.add_argument("--lsa", help="The rank of the SVD approximation.",
                                                                default=100, type=int)
    parser.add_argument("--jobs", help="number of jobs for the grid.",
                                                                default=8, type=int)
    parser.add_argument("--tf", help="TF normalization: none, binary, sublinear (default=none).", default="none")
    parser.add_argument("--stop", help="Toggles stop words stripping.", action="store_true")
    parser.add_argument("--grid", help="Toggles generate a grid of embeddings.", action="store_true")

    args = parser.parse_args()

    if args.grid:
        import itertools
        from joblib import Parallel, delayed
        if args.lsa > 0:
            j=[[20, 50, 100, 200, 300, 400, 500], ["sublinear", "binary", "none"], [True, False]]
            
        else:
            j=[["sublinear", "binary", "none"], [True, False]]

        jobs=itertools.product(*j)
        odir=os.path.normpath(args.cout)
        if not os.path.exists(odir):
            os.makedirs(odir)

        if args.lsa > 0:
            r=Parallel(n_jobs=args.jobs, verbose=10, pre_dispatch='1.5*n_jobs')(delayed(def2tfidf.fit_vectors)(dataset=args.dataset, uniq=True,
                                     lsa=d, tf=t, stop=s, cout = odir + "def2tfidf_H%d_tf-%s_st-%s.vec" % (d, t, s)) for d, t, s in jobs)

        else:
            r=Parallel(n_jobs=args.jobs, verbose=10, pre_dispatch='1.5*n_jobs')(delayed(def2tfidf.fit_vectors)(dataset=args.dataset, uniq=True,
                                     lsa=0, tf=t, stop=s, cout = odir + "def2tfidf_tf-%s_st-%s" % (t, s)) for t, s in jobs)

        print(r)

    else:
    # dataset, cout, tf='binary', stop=True, analyzer='word', lsa=0
        odir=os.path.normpath(args.cout)                
        if not os.path.exists(os.path.dirname(args.cout)):
            os.makedirs(odir)

        r = def2tfidf.fit_vectors(dataset = args.dataset, cout = odir, lsa=args.lsa, tf=args.tf, stop=args.stop, uniq=True)
        print(r)
