import os
import timeit
import numpy,string,re, struct, getopt, sys, time
import theano
import theano.tensor as T
import pprint,math
import six.moves.cPickle as pickle
from readfile import *

theano.config.floatX = 'float32'

def buildModel():
    tmp_emb = T.fmatrix('tmp_emb')
    tmp_theta = T.fmatrix('tmp_theta')
    learning_rate = T.fscalar('learning_rate')
    label = T.ivector()
    index_emd = T.ivector()
    index_theta = T.ivector()
    mid_v1 = T.nnet.sigmoid(T.sum(tmp_emb * tmp_theta, axis=1))
    mid_v = T.clip(mid_v1,1e-7,1-1e-7)
    cost = T.dot(label, T.log(mid_v)) + T.dot((1 - label), T.log(1 - mid_v))
    g_emb = T.grad(cost=cost, wrt=tmp_emb)
    g_theta = T.grad(cost=cost, wrt=tmp_theta)
    # updates = [(emb, emb+learning_rate * g_emb),
    #            (theta, theta+learning_rate * g_theta)]
    updates = [(emb, T.inc_subtensor(emb[index_emd, :], learning_rate * g_emb)),
               (theta, T.inc_subtensor(theta[index_theta, :], learning_rate * g_theta))]
    train_model = theano.function(
        inputs=[index_emd, index_theta, label, learning_rate],
        outputs=cost,
        updates=updates,
        givens={
            tmp_emb: emb[index_emd, :],
            tmp_theta: theta[index_theta, :]},
        allow_input_downcast=True)
    return train_model


def train(filepath,train_line,wdnum,local_iter):
    l_rate = alpha
    wordsPerLine = 5000*cs*2*negative
    train_model = buildModel()
    start_time = timeit.default_timer()
    resW = numpy.zeros((wordsPerLine,), dtype='int32')
    resC = numpy.zeros((wordsPerLine,), dtype='int32')
    mylabel = numpy.zeros((wordsPerLine,), dtype='int32')
    cur_wdnum = 0
    last_wdnum = 0
    meancost = 0
    cn = 1
    if os.path.isfile(filepath):
        print "training..."
        n = 0
        wdnum = wdnum*local_iter
        while local_iter > 0:
            with open(filepath, 'r') as f:
                last_time = timeit.default_timer()
                for line in f:
                    for sen in re.split('[.,;!?:]', line.lower()):
                        tmpsen = re.split('\W+',sen)
                        senNum = 1
                        if len(tmpsen)>1000:
                            tmpsenlist = [tmpsen[x:x + 1000] for x in xrange(0, len(tmpsen), 1000)]
                            senNum = len(tmpsenlist)
                        else:
                            tmpsenlist = list()
                            tmpsenlist.append(tmpsen)
                        for seni in xrange(senNum):
                            tmpsen = tmpsenlist[seni]
                            # record_i = contexwin(tmpsen, resW, resC, mylabel)
                            record_i, sen_length = contexwin_byNumpy(tmpsen, resW, resC, mylabel)
                            cur_wdnum += sen_length
                            if record_i == 0:
                                continue
                            batchNum = record_i // batchSize
                            minibatch_ind = 0
                            for minibatch_ind in range(1,batchNum+1):
                                st = (minibatch_ind-1)*batchSize
                                en = minibatch_ind*batchSize
                                tmpcost = train_model(resW[st:en], resC[st:en], mylabel[st:en], l_rate)
                                meancost += tmpcost
                            if(minibatch_ind*batchSize <= record_i):
                                st = (minibatch_ind) * batchSize
                                en = record_i
                                tmpcost = train_model(resW[st:en], resC[st:en], mylabel[st:en],l_rate)
                                meancost += tmpcost
                            cn += record_i
                            if cur_wdnum - last_wdnum > 10000:
                                l_rate = alpha * (1-float(cur_wdnum)/(wdnum+1))
                                if l_rate < alpha*0.0001:
                                    l_rate = alpha*0.0001
                                cur_time = timeit.default_timer()
                                usedTime = cur_time - start_time
                                sys.stdout.write(
                                    'CostTime:{0:d}s/{1:.2f}h, Alpha:{2:.4f},Words/sec:{3:.1f}k, Progress:{4:.2f}%. Cost:{5:.2f}\r'.format(
                                        int(usedTime), (usedTime) / 3600, l_rate,
                                        10 / (cur_time - last_time),
                                        float(cur_wdnum) * 100 / wdnum,
                                        abs(meancost/cn)
                                    ))
                                last_time = cur_time
                                last_wdnum = cur_wdnum
                                meancost = 0
                                cn = 1
                                sys.stdout.flush()
                    n += 1
                    if n%1000000 == 0:
                        dumpWordVector(emb.get_value(), n)
                        print "\nDump successfully."
                    if n-1 == train_line:
                        print "\nstop training manually."
                        break
            local_iter -= 1
        end_time = timeit.default_timer()
        print "\nWords processed:%d" % cur_wdnum
        print "Word processed per second: %d." % (cur_wdnum / (end_time - start_time))

def contexwin(senlist, resW, resC, mylabel):
    senlist = filter(lambda x: x.isalpha(), senlist)
    senlist = filter(lambda x: x in mapVoca, senlist)
    wlen = len(senlist)
    record_i = 0
    if wlen < 2:
        return record_i
    for loc in xrange(wlen):
        ws = numpy.random.randint(1, cs+1)
        if loc-ws >= 0:
            begin = loc-ws
        else:
            begin = 0
        if wlen-loc > ws:
            en = loc+ws
        else:
            en = wlen-1
        for c in xrange(begin,en+1):
            if c == loc:
                continue
            for i in xrange(negative+1):
                resW[record_i] = mapVoca[senlist[loc]]
                if i == 0:
                    resC[record_i] = mapVoca[senlist[c]]
                    mylabel[record_i] = 1
                else:
                    target = numpy.random.randint(0, len(table))
                    while table[target] == mapVoca[senlist[c]]:
                        target = numpy.random.randint(0, len(table)-1)
                    resC[record_i] = table[target]
                    mylabel[record_i] = 0
                record_i += 1
    return record_i

def contexwin_byNumpy(senlist, resW, resC, mylabel):
    senlist = filter(lambda x: x.isalpha(), senlist)
    record_i = 0
    senlist = filter(lambda x: x in mapVoca, senlist)
    senlength = len(senlist)
    if subSample > 0:
        senlist = filter(lambda x: numpy.random.uniform() < (math.sqrt(sortedVoca[mapVoca[x]][1]/(subSample*wdnum))+1)\
            * (subSample*wdnum)/sortedVoca[mapVoca[x]][1], senlist)
    wlen = len(senlist)
    if wlen < 2:
        return record_i,senlength
    for loc in xrange(wlen):
        ws = numpy.random.randint(1, cs+1)
        if loc-ws >= 0:
            begin = loc-ws
        else:
            begin = 0
        if wlen-loc > ws:
            en = loc+ws
        else:
            en = wlen-1
        P_num = en-begin
        N_num = (en-begin)*negative
        resW[record_i:record_i+P_num+N_num] = mapVoca[senlist[loc]]
        tmpi = 0
        for c in range(begin,en+1):
            if c == loc:
                continue
            resC[record_i+tmpi] = mapVoca[senlist[c]]
            tmpi += 1
        resC[record_i+P_num:record_i + P_num + N_num] = table[numpy.random.randint(0,len(table),(N_num,))]
        mylabel[record_i:record_i+P_num] = 1
        mylabel[record_i+P_num:record_i + P_num + N_num] = 0
        record_i += P_num + N_num
    return record_i,senlength

def dumpWordVector(embvalue,n):
    with open(output_file + "-" + time.strftime('%Y%m%d-%H:%M', time.localtime()) + "+" + str(n) + "L",'wb') as f:
        f.write("%d %d\n" % (nv, de))
        for wi in xrange(nv):
            f.write("%s " % sortedVoca[wi][0])
            for fi in xrange(de):
                if binary == 1:
                    f.write(struct.pack('@1f', embvalue[wi][fi].item()))
                else:
                    f.write("%f " % embvalue[wi][fi].item())


def usage():
    print sys.argv[0] + ' -i trainfile, -o outputfile, -s dimension, -w window, -n negative, -m min_count, -b batchsize, -L trainlines.'
    print sys.argv[0] + ' -h get help info'

if __name__ == '__main__':
    filepath = "./brown.txt"
    output_file = "./word2vec-master/emb.bin"
    table = None
    keepOld = False
    tablesize = 10000000
    de, cs = 200, 5
    negative = 20
    subSample = 0.0001
    min_count = 5
    binary = 1
    local_iter = 1
    batchSize = 128
    alpha = 0.025
    train_line = 10000000
    opts, args = getopt.getopt(sys.argv[1:],"hi:o:s:w:n:m:b:L:")
    for op, value in opts:
        if op == "-i":
            filepath = value
        elif op == "-o":
            output_file = value
        elif op == "-s":
            de = int(value)
        elif op == "-w":
            cs = int(value)
        elif op == "-n":
            negative = int(value)
        elif op == "-m":
            min_count = int(value)
        elif op == "-b":
            batchSize = int(value)
        elif op == "-L":
            train_line = int(value)
        elif op == '-h':
            usage()
            sys.exit()
    print "Dimension:"+str(de)+", Window:"+str(cs)+", Negtive:"+str(negative)+\
          ", Min_count:"+str(min_count)+", Trainlines:"+str(train_line)+", Batchsize:"+str(batchSize)
    print "TrainFile:",filepath
    print time.strftime('%Y%m%d-%H:%M', time.localtime())
    train_line,sortedVoca = getVoca(filepath,keepOld,min_count,train_line)
    mapVoca = initMapIndex(sortedVoca,keepOld)
    wdnum, table = initTable_byNumpy(sortedVoca, tablesize, False)
    print "size of mapVoca is %d" % len(mapVoca)
    nv = len(sortedVoca)
    emb = theano.shared((0.5 / de) * numpy.random.uniform(-1.0, 1.0, (nv,de)).astype(theano.config.floatX))
    theta = theano.shared(numpy.zeros(shape=(nv,de)).astype(theano.config.floatX))
    print "building model..."
    begp = emb.get_value()
    train(filepath,-1,wdnum,local_iter)
    endp = emb.get_value()
    dumpWordVector(emb.get_value(), train_line)
    print "dump parameter successfully."
    print numpy.linalg.norm(endp - begp)

