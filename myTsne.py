import numpy as np
import os,numpy, struct
from matplotlib import pyplot as plt
from tsne import bh_sne
import six.moves.cPickle as pickle
import theano.tensor as T
import theano
from matplotlib.pyplot import savefig
# emb = pickle.load(open('./tmp/emb.txt'))
with open('./word2vec-master/emb.bin-20160719-14:12+19810L','rb') as f:
    nv_and_de = f.readline().split()
    nv = int(nv_and_de[0])
    de = int(nv_and_de[1])
    data = numpy.zeros((nv,de),dtype=numpy.float)
    vocabulary = list()
    ri = 0
    while ri < nv:
        ch = f.read(1)
        tmpword = ""
        while ch != " ":
            tmpword += ch
            ch = f.read(1)
        vocabulary.append(tmpword)
        di = 0
        while di<de:
            bytes = f.read(4)
            tf,=struct.unpack('@1f', bytes)
            data[ri][di] = tf
            di += 1
        ri += 1
print data.shape
print type(data)
shownum = 100
x2d = bh_sne(data[:shownum,:].astype(numpy.float64))
print x2d.shape
dotter = plt.figure(figsize=(32,18))
plt.plot(x2d[:shownum,0],x2d[:shownum,1],'.')
font = {
    'weight':'normal',
    'size': 8
}
for i in range(0,shownum):
    plt.text(x2d[i,0],x2d[i,1],vocabulary[i],fontdict=font)
savefig('wordvec.eps',format='eps',dpi=1600)
print "over"