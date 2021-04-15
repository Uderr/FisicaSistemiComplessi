import numpy as np
import config as conf
import time
import matplotlib.pyplot as plt
import random


tempoIniziale = time.time()

numCitt = 40
numeroIndividui = 50
numeroGenerazioni = 500
probMutazione = 0.25


def copiaLista(vecchioArr: [int]):
    nuovoArr = []
    for element in vecchioArr:
        nuovoArr.append(element)
    return nuovoArr


class Cromosomi:
    def __init__(self, gene = None):
        #Inizializzazione random di un gene
        if gene is None:
           gene = [i for i in range(numCitt)]
           random.shuffle(gene)
        self.gene = gene
        self.fitness = self.calcolaFitness()

    def calcolaFitness(self):
        fitness = 0.0
        #Calcola fitness individui
        for i in range(numCitt - 1):
            #Città partenza e di arrivo
            daInd = self.gene[i]
            aInd = self.gene[i + 1]
            fitness += matriceDistCitt[daInd, aInd]
        #Connette inizio a fine
        fitness += matriceDistCitt[self.gene[-1], self.gene[0]]
        return fitness


class algoGen:
    def __init__(self, input):
        global matriceDistCitt
        matriceDistCitt = input
        self.migliore = None #Migliore individuo
        self.listaIndividui = [] #Lista individui
        self.soluzioniGen = [] #Soluzione generazione
        self.listaFitness = [] #Fitness generazione

    def crossover(self):
        nuovaGenerazione = []
        random.shuffle(self.listaIndividui)
        for i in range(0, numeroIndividui - 1, 2):
            #Gene paterno
            gene1 = copiaLista(self.listaIndividui[i].gene)
            gene2 = copiaLista(self.listaIndividui[i + 1].gene)
            indice1 = random.randint(0, numCitt - 2)
            indice2 = random.randint(indice1, numCitt - 1)
            posizione1Reco = {value: idx for idx, value in enumerate(gene1)}
            posizione2Reco = {value: idx for idx, value in enumerate(gene2)}
            #Crossover
            for j in range(indice1, indice2):
                value1, value2 = gene1[j], gene2[j]
                pos1, pos2 = posizione1Reco[value2], posizione2Reco[value1]
                gene1[j], gene1[pos1] = gene1[pos1], gene1[j]
                gene2[j], gene2[pos2] = gene2[pos2], gene2[j]
                posizione1Reco[value1], posizione1Reco[value2] = pos1, j
                posizione2Reco[value1], posizione2Reco[value2] = j, pos2
            nuovaGenerazione.append(Cromosomi(gene1))
            nuovaGenerazione.append(Cromosomi(gene2))
        return nuovaGenerazione

    def mutazione(self, nuovaGenerazione):
        for individuo in nuovaGenerazione:
            if random.random() < probMutazione:
                vecchiaGenerazione = copiaLista(individuo.gene)
                indice1 = random.randint(0, numCitt - 2)
                indice2 = random.randint(indice1, numCitt - 1)
                geneMutato = vecchiaGenerazione[indice1:indice2]
                geneMutato.reverse()
                individuo.gene = vecchiaGenerazione[:indice1] + geneMutato + vecchiaGenerazione[indice2:]
        #Unire due generazioni
        self.listaIndividui += nuovaGenerazione

    def seleziona(self):
        numeroGruppi = 10 #Numero gruppi
        grandezzaGruppo = 10 #Numero individui gruppo
        numeroVincitori = numeroIndividui // numeroGruppi #Numero vincitori gruppo
        vincitori = [] #Risultato
        for i in range(numeroGruppi):
            gruppo = []
            for j in range(grandezzaGruppo):
                #Gruppo randomico
                selezionato = random.choice(self.listaIndividui)
                selezionato = Cromosomi(selezionato.gene)
                gruppo.append(selezionato)
            gruppo = algoGen.rank(gruppo)
            #Seleziona vincitore
            vincitori += gruppo[:numeroVincitori]
        self.listaIndividui = vincitori

    def rank(gruppo):
        #Bubble Sort
        for i in range(1, len(gruppo)):
            for j in range(0, len(gruppo) - i):
                if gruppo[j].fitness > gruppo[j + 1].fitness:
                    gruppo[j], gruppo[j + 1] = gruppo[j + 1], gruppo[j]
        return gruppo

    def nuovaGenerazione(self):
        #Crossover
        nuovaGenerazione = self.crossover()
        #Mutazione
        self.mutazione(nuovaGenerazione)
        #Seleziona
        self.seleziona()
        #Risultati per la generazione
        for individuo in self.listaIndividui:
            if individuo.fitness < self.migliore.fitness:
                self.migliore = individuo

    def train(self):
        #Popolazione primaria
        self.listaIndividui = [Cromosomi() for _ in range(numeroIndividui)]
        self.migliore = self.listaIndividui[0]
        #Iterazione
        for i in range(numeroGenerazioni):
            self.nuovaGenerazione()
            risultato = self.migliore.gene
            #Connetti inizio a fine
            risultato.append(risultato[0])
            self.soluzioniGen.append(risultato)
            self.listaFitness.append(self.migliore.fitness)
        return self.soluzioniGen, self.listaFitness



def matriceDistanze(listaInput):
    n = numCitt
    distMatr = np.zeros([n, n])
    for i in range(n):
        for j in range(i + 1, n):
            d = listaInput[i, :] - listaInput[j, :]
            #Calcola prodotto
            distMatr[i, j] = np.dot(d, d)
            distMatr[j, i] = distMatr[i, j]
    return distMatr


def main():

    #Coordinate città
    coordinateCitt = np.random.rand(numCitt, 2)

    #Matrice distanze città
    matriceDistCitt = matriceDistanze(coordinateCitt)


    #Operazioni algoritmi genetici
    algoriGen = algoGen(matriceDistCitt)
    soluzioniGen, listaFitness = algoriGen.train()
    risultato = soluzioniGen[-1]
    risultatoCoordinate = coordinateCitt[risultato, :]



    #Stampa grafici
    plt.plot(risultatoCoordinate[:, 0], risultatoCoordinate[:, 1],  'o-r')
    plt.show()

    plt.figure()
    plt.plot(listaFitness)
    plt.xlabel("Generazioni")
    plt.ylabel("Distanza percorsa")
    plt.show()

    print("La distanza inizialmente percorsa è: " + str(listaFitness[0]))
    print("La distanza percorsa dall'ultima generazione è: " + str(listaFitness[-1]))

    print("--- %s secondi ---" % (time.time() - tempoIniziale))

if __name__ == '__main__':
    main()