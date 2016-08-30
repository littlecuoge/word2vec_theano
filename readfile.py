import os
import pprint,numpy,math,re
import six.moves.cPickle as pickle

def getVoca(filepath,keepOld,min_count,train_line):
    if os.path.isfile('./tmp/voca.txt') and keepOld:
        f = open('./tmp/voca.txt','r')
        print "the vocabulary file already exists."
        NmapVoca = pickle.load(f)
        n = NmapVoca[0]
        voca = NmapVoca[1]
    else:
        n = 0
        print "build new vocabulary..."
        f = open(filepath,'r')
        voca2 = dict()
        for line in f:
            # for word in line.split():
            for word in re.split('\W+', line.lower()):
                if not word.isalpha():
                    continue
                if voca2.has_key(word):
                    voca2[word]=voca2[word]+1
                else:
                    voca2[word]=1
            n += 1
            if (n%1000000)==0:
                print "%d lines have been read." %n
            if n == train_line:
                break
        print "read lines is:%d" %n
        f.close()
        voca = sorted(voca2.items(),lambda x,y:cmp(x[1],y[1]),reverse = True)
        if min_count > 0:
            beg = 0
            en = len(voca)-1
            while(beg < en):
                mid = beg + (en-beg)/2
                if voca[mid][1] == min_count:
                    beg = mid
                    break
                elif voca[mid][1] > min_count:
                    beg = mid+1
                else:
                    en = mid-1
            voca = voca[:beg+1]
        voca.insert(0,('</s>',999))
        f = open('./tmp/voca.txt', 'w')
        NmapVoca = (n,voca)
        pickle.dump(NmapVoca,f)
    f.close()
    print "vocabulary size:%d" %len(voca)
    return n,voca #line number and sorted vocabulary


def initMapIndex(sortedVoca, keepOld):
    if os.path.isfile('./tmp/mapVoca.txt') and keepOld:
        f=open('./tmp/mapVoca.txt','r')
        print "map file already exists"
        mapVoca = pickle.load(f)
    else:
        print "build new mapfile..."
        mapVoca = dict()
        for i in xrange(len(sortedVoca)):
            mapVoca[sortedVoca[i][0]] = i
        f=open('./tmp/mapVoca.txt','w')
        pickle.dump(mapVoca,f)
    f.close()
    return mapVoca

def initTable(sortedVoca,tablesize,keepOld):
    if os.path.isfile('./tmp/table.txt') and keepOld:
        f=open('./tmp/table.txt','r')
        print "table file already exists"
        tableMapwdnum = pickle.load(f)
        wdnum = tableMapwdnum[0]
        table = tableMapwdnum[1]
    else:
        print "build new table..."
        power = 0.75
        # tablesize = 100000
        table = [0]*tablesize
        totalPow = 0
        wdnum = 0
        for wd in sortedVoca:
            totalPow += math.pow(wd[1],power)
            wdnum += wd[1]
        i=0
        print "total word num:%d" %wdnum
        print "total pow is:%f" %totalPow
        d1 = math.pow(sortedVoca[0][1],power) / totalPow
        for a in xrange(0,tablesize):
            table[a] = i
            if ((float(a)/tablesize) > d1):
                i += 1
                d1 += math.pow(sortedVoca[i][1],power)/totalPow
            if i>tablesize:
                i = tablesize - 1
            # if a%40000 == 0:
            #     print "table inited cpmpleted:%.2f %%" %(float(a)*100/tablesize)
        f=open('./tmp/table.txt','w')
        tableMapwdnum = (wdnum, table)
        pickle.dump(tableMapwdnum,f)
    f.close()
    print "table size:%d" %len(table)
    return wdnum,table


def initTable_byNumpy(sortedVoca,tablesize,keepOld):
    if os.path.isfile('./tmp/py_table.txt') and keepOld:
        with open('./tmp/py_table.txt','r') as f:
            print "table file already exists"
            tableMapwdnum = pickle.load(f)
            wdnum = tableMapwdnum[0]
            py_table = tableMapwdnum[1]
    else:
        print "build new py_table..."
        power = 0.75
        py_table = numpy.zeros((tablesize,),dtype='int32')
        totalPow = 0
        wdnum = 0
        for wd in sortedVoca:
            totalPow += math.pow(wd[1],power)
            wdnum += wd[1]
        i=0
        print "total word num:%d" %wdnum
        d1 = math.pow(sortedVoca[0][1],power) / totalPow
        for a in xrange(0,tablesize):
            py_table[a] = i
            if ((float(a)/tablesize) > d1):
                i += 1
                d1 += math.pow(sortedVoca[i][1],power)/totalPow
            if i>tablesize:
                i = tablesize - 1
        # f=open('./tmp/py_table.txt','w')
        # tableMapwdnum = (wdnum, py_table)
        # pickle.dump(tableMapwdnum,f)
        # f.close()
    print "table size:%d" %len(py_table)
    return wdnum,py_table

if __name__ == '__main__':
    keepOld = False
    linenum, sortedVoca = getVoca(filepath, keepOld, -1)
    mapVoca = initMapIndex(sortedVoca,keepOld)
    print "size of sortedVoca is%d" %len(sortedVoca)
    print "size of mapVoca is %d" %len(mapVoca)
    wdnum, table = initTable(sortedVoca,100000000,keepOld)







