rd=$1

for d in `ls -d ${rd}/*_l-*`; do python evaluation.py $d $d; done
printf "Method\t\\lambda\t\\\alpha\tF1-score\n" > ${rd}/results.csv
for d in `ls -d ${rd}/*_l-*`; do 
	printf "%s\t%s\n" $(basename $d) $(awk -F':' '{print $2}' $d/scores.txt)
done |
	sed 's/\(_l\-\)/\t/g' | sed 's/\(_h\-\)/\t/g' |
	sort -k4 -r>> ${rd}/results.csv

