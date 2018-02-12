d=$1 &&\
D=results_$d &&\
mkdir $D &&\
cd $D &&\
python ../attributes.py --embeddings ../../def2tfidf_grid/$d.bin --queries ~/DiscriminAtt/training/validation.txt --bin --normalize --grid --jobs -1 &&\
cd .. &&\
bash tabulate.sh $D &&\
for d in `ls -d results_*`; do echo ${d}; head -n4 ${d}/results.csv; done>attr_results.all &&\
more attr_results.all
