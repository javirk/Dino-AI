import collections
from collections import OrderedDict
import numpy as np
import tensorflow as tf
from perceptron_2l import Perceptron
from Dino import Game
from Dino import Dino
import random

import logging

logger = logging.getLogger('genome')


class Genome(object):
    def __init__(self, numGenes, mutationProb, selection, folder, generations):
        if selection > numGenes:
            self.numGenes = numGenes
            self.selection = numGenes
        else:
            self.numGenes = numGenes
            self.selection = selection

        self.mutationProb = mutationProb
        self.genes = []
        self.generation = 0
        self.only_mutation = 0.2
        self.scores = []
        self.folder = folder
        self.num_Generations = generations
        self.delta = 0.1*self.num_Generations/(self.generation+1)

        self.build_genome()

    #Build the network
    def build_genome(self):
        while len(self.genes) < self.numGenes:
            print(len(self.genes))
            network = Perceptron(self.folder, len(self.genes))
            self.genes.append(network)
            network = None


    def execute_generation(self, game):
        self.gen = 0
        self.scores = []
        self.generation += 1
        self.delta = 0.1 * self.num_Generations / self.generation

        while self.gen < self.numGenes:
            self.execute_gene(game)
            self.gen += 1

    def execute_gene(self, game):
        gene = self.genes[self.gen]

        #Si está parado, reinicia
        if game.get_crashed():
            game.restart()

        dinosaur = Dino(game)

        #current_score = dinosaur.play(gene)
        self.genes[self.gen].fitness = dinosaur.play(gene, self.gen, self.generation, self.folder)
        #self.scores.append(current_score)

    def select_best_genes(self):
        d = dict(enumerate(self.genes))

        f = []
        s = []
        selected = OrderedDict(sorted(d.items(), key=lambda t: t[1].fitness, reverse=True)).values()
        selected_list = list(selected)
        selected = selected_list[:self.selection]
        tf.reset_default_graph()
        for select in selected:
            fit = select.copy()
            fit.reload()
            s.append(fit)
            f.append(select.fitness)

        selected = None
        logger.info('Fitness: #### %s' % (str(f),))
        return s, f

    def kill_and_reproduce(self):
        #Kill
        self.genes, scores = self.select_best_genes()
        logger.info("Renumbering...")
        #Renumber
        for gen in self.genes:
            gen.n_gen = self.genes.index(gen)

        logger.info(f"{self.selection} genes have been renumbered.")

        best = self.genes.copy()

        prob = self.get_probability_distribution(scores)

        #Crossover
        while len(self.genes) < self.numGenes - int(self.only_mutation*self.numGenes):
            genA = random.choices(best, k=1, weights = prob)[0].copy()
            genB = random.choices(best, k=1, weights = prob)[0].copy()

            newGen = self.mutation(self.crossover(genA, genB))
            newGen.n_gen = len(self.genes)
            self.genes.append(newGen)

        #Mutation
        while len(self.genes) < self.numGenes:
            gen = random.choices(best, k=1, weights = prob)[0].copy()

            newGen = self.mutation(gen)
            newGen.n_gen = len(self.genes)
            self.genes.append(newGen)

        logger.info(f'Generación completada {self.generation}')

    def get_probability_distribution(self, scores):
        p = [i / np.sum(scores) for i in scores]
        logger.info(f'The probabilities are {p}')
        return p

    def crossover(self, gen1, gen2):
        if random.random() > 0.5:
            temp = gen1
            gen1 = gen2
            gen2 = temp

        try:
            gen1_dict = gen1.as_dict
        except:
            gen1_dict = gen1.get_dict

        try:
            gen2_dict = gen2.as_dict
        except:
            gen2_dict = gen2.get_dict


        for par in ['biases']:
            gen1_par = gen1_dict[par]
            gen2_par = gen2_dict[par]
            cut_loc = int(len(gen1_par) * random.random())

            gen1_updated = np.append(gen1_par[(range(0, cut_loc)),], gen2_par[(range(cut_loc, len(gen2_par))),])

            gen1_dict[par] = gen1_updated

        gen1.reload()

        return gen1

    def mutation(self, gen1):
        gen1_dict = gen1.as_dict
        self.mutate_data(gen1_dict, self.mutationProb, 'biases')
        self.mutate_data(gen1_dict, self.mutationProb, 'weights')
        gen1.reload()

        return gen1

    def mutate_data(self, gen_dict, mutationRate, key):
        for k in range(0,  len(gen_dict[key])):
            if random.random() < mutationRate:
                #gen_dict[key][k] += random.uniform(-self.delta, self.delta)
                gen_dict[key][k] += gen_dict[key][k] * (random.random() - 0.5) * random.random() + (random.random() - 0.5)

    def save_all(self):
        logger.info("Saving all genomes.")
        for gen in self.genes:
            #gen.saver = tf.train.Saver()
            gen.save_net()
