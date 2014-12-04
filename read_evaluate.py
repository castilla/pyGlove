__author__ = 'castilla'
import numpy as np
import operator
import csv


if __name__ == "__main__":
    palavras=[]
    ocorrencias=[]
    vocab = open('../vocab.txt', 'r')
    lista = vocab.readlines()
    for item in lista:
        partes = item.split(" ")
        palavras.append(partes[0])
        ocorrencias.append(int(partes[1].strip()))
    order = range(0,len(palavras)+1)
    wordMap = dict(zip(palavras, order))
    sorted_wordMap = sorted(wordMap.items(), key=operator.itemgetter(1))
    vocab_size = len(palavras)
    vetores = open('../vectors.bin', 'rb')
    vetores.seek(0, 2)
    vector_size = (vetores.tell()/16/vocab_size)-1
    vetores.seek(0, 0)
    with vetores as fid:
        data_array = np.fromfile(fid).reshape((-1, vector_size+1))
    data_array = np.delete(data_array, np.s_[-1:], 1)
    data_arrays = np.split(data_array, 2)
    W1 = data_arrays[0]
    W2 = data_arrays[1]
    W = np.add(W1, W2)
    Wf = W / np.sqrt(np.sum(W*W, axis=1))[:,np.newaxis]

    filenames = ['capital-common-countries']


    '''

            ,'capital-world', 'currency' , 'city-in-state' ,
             'family', 'gram1-adjective-to-adverb','gram2-opposite' ,'gram3-comparative',
             'gram4-superlative', 'gram5-present-participle' , 'gram6-nationality-adjective'
             ,'gram7-past-tense', 'gram8-plural', 'gram9-plural-verbs']
    '''

    path = './question-data/'

    split_size = 100
    correct_sem = 0
    correct_syn = 0
    correct_tot = 0
    count_syn = 0
    count_tot = 0
    full_count = 0

    for arquivo in filenames:
        full_arquivo = path + arquivo+".txt"
        ind1 = []
        ind2 = []
        ind3 = []
        ind4 = []
        inds = ind1, ind2, ind3, ind4

        with open(full_arquivo, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
            #for row in reader:
                itens = row[0].split(" ")
                i = 0
                for item in inds:
                    try:
                        item.append(wordMap[itens[i]])
                    except KeyError:
                        item.append(float('-inf'))
                    i += 1

        fullsize = len(ind1)

        ind1 = np.asarray(ind1)
        ind2 = np.asarray(ind2)
        ind3 = np.asarray(ind3)

        mx = np.zeros((1, fullsize))
        num_iter = int(np.ceil(float(fullsize)/split_size))

        print num_iter
        for num in range(1, num_iter+1):
            x = range((num-1)*split_size, min((num)*split_size, fullsize))
            dist = np.dot(Wf, (Wf[ind2[x], :].T - Wf[ind1[x], :].T + Wf[ind3[x], :].T))
            for i in range(0,len(x)):
                dist[ind1[x[i]],i] = float('-inf')
                dist[ind2[x[i]],i] = float('-inf')
                dist[ind3[x[i]],i] = float('-inf')
            # [~, mx(range)] estamos aqui


        v, z = dist.max(0), dist.argmax(0)
        print v, z
        for item4 in z:
            print wordMap.keys()[wordMap.values().index(item4)]




    pass
