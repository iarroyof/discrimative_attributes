from joblib import Parallel, delayed
import argparse, codecs

from pdb import set_trace as st

def rm_words(user_input, stop_words):
    """Sanitize using intersection and list.remove()"""
    # Downsides:
    #   - Looping over list while removing from it?
    #     http://stackoverflow.com/questions/1207406/remove-items-from-a-list-while-iterating-in-python

    stop_words = set(stop_words)
    for sw in stop_words.intersection(user_input):
        while sw in user_input:
            user_input.remove(sw)

    return user_input

def keep_words(input_line, words_kept, count_dict, limit, stop, out_file):
    if input_line == []: return None
    if stop: input_line=rm_words(input_line, stop)
   
    with codecs.open(out_file, mode = "a", encoding = 'latin-1', errors = 'replace') as f:
        already=False
        for w in words_kept:
            try:
                
                if count_dict[w] > limit:
                    words_kept.pop(words_kept.index(w))
                    print("Left to find %d\n" % len(words_kept))
                    continue

                if w in input_line:
                    count_dict[w]+=1
                    if not already:
                        f.write("%s\n" %  " ".join(input_line))
                    already=True
                else:
                    continue

            except KeyError:
                if w in input_line:
                    count_dict[w]=1
                    if not already:
                        f.write("%s\n" %  " ".join(input_line))
                    already=True
                else:
                    continue


class streamer(object):
    def __init__(self, file_name):
        self.file_name=file_name

    def __iter__(self):
       for ln in codecs.open(self.file_name, mode = "r", encoding = 'latin-1', errors = 'replace'):
           yield ln.strip().lower().split()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Computes attribute sharing degree between word vectors.')
    parser.add_argument("--input", help = "The path to the input text file.", required = True)
    parser.add_argument("--need", help = "The path to the needed words file.", required = True)
    parser.add_argument("--stop", help = "The path to stop words file.", default=None)
    parser.add_argument("--fmax", help = "The maximum word occurrences.", type = int, default=5)
    parser.add_argument("--output", help = "Filtered output file.", default = "filtered_file.txt")
    args = parser.parse_args()

    ls=[]
    for ln in codecs.open(args.need, mode = "r", encoding = 'latin-1', errors = 'replace'):
        ls = ls + ln.strip().lower().split(",")

    ls=[x for x in set(ls) if not isinstance(x, int)]

    if args.stop:
        with codecs.open(args.stop, mode = "r", encoding = 'latin-1', errors = 'replace') as f:
            stopwords = f.read().strip().split('\n')
    else:
        stopwords = None

    dic={}

    gen_file=streamer(args.input)

    # input_line, words_kept, count_dict, limit, stop, out_file
    #Parallel(n_jobs=20, verbose=5)(delayed(keep_words)(i, ls, dic, args.fmax, stopwords, 
    #                                                                 args.output) for i in gen_file)
    for i in gen_file:
        keep_words(i, ls, dic, args.fmax, stopwords, args.output)
    
#    print("Found %d / %d" % (total, len(ls)))
