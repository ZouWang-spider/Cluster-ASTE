import nltk
import dgl
from nltk import Tree
import numpy as np
import torch
from supar import Parser
from stanfordcorenlp import StanfordCoreNLP



#Using Stanford Parsing to get Word and Part-of-speech
# nlp = StanfordCoreNLP(r'D:/StanfordCoreNLP/stanford-corenlp-4.5.4', lang='en')
parser = Parser.load('biaffine-dep-en')


# # load BERT model and Tokenizer
# model_name = 'D:/bert-base-cased'
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name)
#
# #use Stanford NLP for constituent tree
# def constituent_tree(sentence):
#     tree_str = nlp.parse(sentence)
#     constituent_tree = Tree.fromstring(tree_str)
#
#     return constituent_tree



def Dependency_relation(sentence):
      text = nltk.word_tokenize(sentence)

      # biaffine for dependency relationship
      dataset = parser.predict([text], prob=True, verbose=True)
      # print(dataset.sentences[0])
      relation = dataset.rels[0]
      # print(f"arcs:  {dataset.arcs[0]}\n"
      #       f"rels:  {dataset.rels[0]}\n"
      #       f"probs: {dataset.probs[0].gather(1,torch.tensor(dataset.arcs[0]).unsqueeze(1)).squeeze(-1)}")

      # Bulid Graph Using DGL
      arcs = dataset.arcs[0]  # edge information
      edges = [i + 1 for i in range(len(arcs))]
      for i in range(len(arcs)):
            if arcs[i] == 0:
                  arcs[i] = edges[i]

      # Serial number minus one, starting with serial number 0
      arcs = [arc - 1 for arc in arcs]
      edges = [edge - 1 for edge in edges]
      graph = (arcs, edges)
      syn_graph = torch.tensor(graph)  #tensor

      # Initialize the adjacency matrix（float）
      adj_matrix = np.zeros((len(arcs), len(arcs)), dtype=float)
      # Filling the adjacency matrix
      for start, end in zip(arcs, edges):
          adj_matrix[start, end] = 1.0
          adj_matrix[end, start] = 1.0  # If it is an undirected graph, the symmetric edges need to be added as well


      return relation, graph



###########################Test#################################
# sentence = "The body is a bit cheaply made so it will be interesting to see how long it holds up ."
#
# relation, graph = Dependency_relation(sentence)
# print(relation)  #['det', 'nn', 'nsubj', 'cop', 'root', 'punct']
# print(graph)  #([2, 2, 4, 4, 4, 4], [0, 1, 2, 3, 4, 5])

