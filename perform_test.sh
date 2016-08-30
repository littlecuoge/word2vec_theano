#!/usr/bin/env bash
#././word2vec-master/word2vec -train ./brown.txt -output ./word2vec-master/vectors.bin -cbow 0 -size 200 -window 5 -negative 10 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 1
python noSubWord2vec.py -i ./text8 -o ./word2vec-master/emb -s 200 -w 5  -n 10 -m 5 -b 1024


# -i trainfile, -o outputfile, -s dimension, -w window, -n negative, -m min_count, -b batchsize, -L trainlines