#from vocabulary.vocabulary import Vocabulary as vb
from PyDictionary import PyDictionary as Dictionary
import re, ast
#from nltk.stem.wordnet import WordNetLemmatizer
import re, html.parser
from joblib import Parallel, delayed


#lemmatizer = WordNetLemmatizer()
output_corpus="/home/iarroyof/DiscriminAtt/forgoten_2.defs"
hash="/home/iarroyof/DiscriminAtt/forgoten2.dict"
vb = Dictionary()
maxd = 2

def get_meanings(word, f):
    #syns=vb.meaning(lemmatizer.lemmatize(word.strip()))
    syns=vb.meaning(word.strip())
    if syns is not None:
        #syns = ast.literal_eval(syns)
        #syns=[html.parser.HTMLParser().unescape(d["text"]) for d in syns if not re.findall("<i>(.*?)\</i>", d["text"])]
        try:
            nouns = [d  for i, d in enumerate(syns['Noun']) if i <= maxd]
        except KeyError:
            nouns = []

        try:
            verbs = [d  for i, d in enumerate(syns['Verb']) if i <= maxd]
        except KeyError:
       	    verbs = []
        #for d in syns:
        f.write("%s\t%s\n" %  (word.strip(), "; ".join(nouns + verbs).lower()))
    else:
        print("not_found\t%s" % word.strip())
        return None


with open(hash, "r") as fh, open(output_corpus, "a+", encoding = 'utf-8', errors = 'replace') as f:
    #Parallel(n_jobs=1)(delayed(get_meanings)(w, f) for w in fh)
    for word in fh:
        get_meanings(word, f)
