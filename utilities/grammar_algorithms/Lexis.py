
# NOTE that this is not my code and it was taken from https://github.com/payamsiyari/Lexis/blob/master/Lexis.py




# -*- coding: utf-8 -*-
"""
@author: Payam Siyari
"""
from __future__ import division
import os
import random
from bisect import bisect_left
import fileinput
import sys
import getopt
import operator
import time
import subprocess
import networkx as nx

class SequenceType:
    Character, Integer, SpaceSeparated = ('c', 'i', 's')
class CostFunction:
    ConcatenationCost, EdgeCost = ('c', 'e')
class RepeatClass:
    Repeat, MaximalRepeat, LargestMaximalRepeat, SuperMaximalRepeat = ('r', 'mr', 'lmr', 'smr')
class LogFlag:
    ConcatenationCostLog, EdgeCostLog = range(2)

class DAG(object):
    __preprocessedInput = [] #Original input as a sequence of integers
    __dic = {} #Dictionary for correspondence of integers to original chars (only when charSeq = 'c','s')
    __DAG = {} #Adjacency list of DAG
    __DAGGraph = nx.MultiDiGraph()
    __DAGStrings = {}#Strings corresponding to each node in DAG

    __concatenatedDAG = [] #Concatenated DAG nodes with seperatorInts
    __concatenatedNTs = [] #For each DAG node, alongside the concatenated DAG
    __separatorInts = set([]) #Used for seperating DAG nodes in the concatenatedDAG
    __separatorIntsIndices = set([]) #Indices of separatorInts in the concatenated DAG
    __nextNewInt = 0 #Used for storing ints of repeat symbols and separators in odd numbers

    __quietLog = False #if true, disables logging
    __iterations = 0

    def __init__(self, inputFile, loadDAGFlag, chFlag = SequenceType.Character, noNewLineFlag = True):
        if loadDAGFlag:
            self.__initFromDAG(inputFile)
        else:
            self.__initFromStrings(inputFile, chFlag, noNewLineFlag)
    #Initializes (an unoptimized) DAG from inputFile. charSeq tells if inputFile is a char sequence, int sequence or space-separated sequence
    def __initFromStrings(self, inputFile, chFlag = SequenceType.Character, noNewLineFlag = True):
        (self.__preprocessedInput, self.__dic) = self.__preprocessInput(inputFile, charSeq = chFlag, noNewLineFlag = noNewLineFlag)
        allLetters = set(map(int,self.__preprocessedInput.split()))
        #Setting odd and even values for __nextNewInt and __nextNewContextInt
        self.__nextNewInt = max(allLetters)+1
        if self.__nextNewInt % 2 == 0:
            self.__nextNewInt += 1
        #Initializing the concatenated DAG
        for line in self.__preprocessedInput.split('\n'):
            line = line.rstrip('\n')
            self.__concatenatedDAG.extend(map(int,line.split()))
            self.__concatenatedDAG.append(self.__nextNewInt)
            self.__concatenatedNTs.extend(0 for j in range(len(map(int,line.split()))))
            self.__concatenatedNTs.append(self.__nextNewInt)
            self.__separatorInts.add(self.__nextNewInt)
            self.__separatorIntsIndices.add(len(self.__concatenatedDAG)-1)
            self.__nextNewInt += 2
    #Loads the DAG from an external file (The file should start from 'N0' line, without cost logs)
    def __initFromDAG(self, inputFile):
        textFile = inputFile.read().splitlines()
        maxInt = -1
        for line in textFile:
            nt = int(line.split(' ->  ')[0][1:])
            self.__dic[nt] = nt
            rhs = line.split(' ->  ')[1].split()
            for w in rhs:
                # sys.stderr.write(w + "\n")
                try:
                    word = int(w)
                except:
                    word = int(w[1:])
                if maxInt < word:
                    maxInt = word
                self.__dic[word] = word
                self.__concatenatedDAG.append(word)
                self.__concatenatedNTs.append(nt)
            self.__concatenatedDAG.append(-1)
            self.__concatenatedNTs.append(-1)
            self.__separatorIntsIndices.add(len(self.__concatenatedDAG) - 1)
        self.__nextNewInt = maxInt + 1
        for i in self.__separatorIntsIndices:
            self.__concatenatedDAG[i] = self.__nextNewInt
            self.__concatenatedNTs[i] = self.__nextNewInt
            self.__separatorInts.add(self.__nextNewInt)
            self.__nextNewInt += 1
        # wordDict = {}
        # counterDict = {}
        # counter = 0
        # textFile = inputFile.read().splitlines()
        # tmpnode = []
        # for line in textFile:
        #     # if len(line.split(' ->  ')) < 2:
        #     #     tmpnode = ['\n'] + line.split(' ')
        #     #     newnode = []
        #     #     for w in tmpnode:
        #     #         if w not in counterDict:
        #     #             wordDict[counter] = w
        #     #             counterDict[w] = counter
        #     #             counter += 1
        #     #         newnode.append(counterDict[w])
        #     #     self.__DAG[newNt] += newnode
        #     #     continue
        #     # else:
        #     nt = int(line.split(' ->  ')[0][1:])
        #     if counter % 2 == 0:
        #         if counter != 0:
        #             counter += 1
        #     if nt not in counterDict:
        #         wordDict[counter] = nt
        #         counterDict[nt] = counter
        #         counter += 1
        #     newNt = counterDict[nt]
        #     node = line.split(' ->  ')[1].split(' ')
        #     newnode = []
        #     for w in node:
        #         if w[0] == 'N':
        #             if w not in counterDict:
        #                 wordDict[counter] = w[1:]
        #                 counterDict[w[1:]] = counter
        #                 counter += 1
        #             newnode.append(counterDict[w[1:]])
        #         else:
        #             if w not in counterDict:
        #                 wordDict[counter] = w
        #                 counterDict[w] = counter
        #                 counter += 1
        #             newnode.append(counterDict[w])
        #     if newNt == 0:
        #         if newNt in self.__DAG:
        #             self.__DAG[newNt].append(newnode)
        #         else:
        #             self.__DAG[newNt] = [newnode]
        #     else:
        #         self.__DAG[newNt] = newnode
        # self.__dic = wordDict
        # self.__nextNewInt = counter
        # if self.__nextNewInt % 2 == 0:
        #     self.__nextNewContextInt = self.__nextNewInt
        #     self.__nextNewInt += 1
        # else:
        #     self.__nextNewContextInt = self.__nextNewInt + 1
        # for nt in self.__DAG:
        #     self.__concatenatedDAG.extend(self.__DAG[nt])
        #     self.__concatenatedDAG.append(self.__nextNewInt)
        #     self.__concatenatedNTs.extend(nt for j in range(len(self.__DAG[nt])))
        #     self.__concatenatedNTs.append(self.__nextNewInt)
        #     self.__separatorInts.add(self.__nextNewInt)
        #     self.__separatorIntsIndices.add(len(self.__concatenatedDAG)-1)
        #     self.__nextNewInt += 2
        # print self.__DAG
        # print self.__dic
        self.__createAdjacencyList()
        # print 'self dag'
        # print self.__DAG
        self.__createDAGGraph()
        # print 'self graph'
        # print self.__DAGGraph
        # print self.__DAGGraph.nodes()
        # print self.__DAGGraph.edges()
        self.__nodeStringsGenerate()
        # print 'self strings'
        # print self.__DAGStrings

    #...........Main G-Lexis Algorithm Functions........
    def GLexis(self, quiet, normalRepeatType, costFunction):
        self.__quietLog = quiet
        while True: #Main loop
            #Logging DAG Cost
            self.__logViaFlag(LogFlag.ConcatenationCostLog)
            self.__logViaFlag(LogFlag.EdgeCostLog)

            #Extracting Maximum-Gain Repeat
            (maximumRepeatGainValue, selectedRepeatOccs) = self.__retreiveMaximumGainRepeat(normalRepeatType, CostFunction.EdgeCost)
            if maximumRepeatGainValue == -1:
                break #No repeats, hence terminate

            self.__logMessage('maxR ' + str(maximumRepeatGainValue) + ' : ' + str(self.__concatenatedDAG[selectedRepeatOccs[1][0]:selectedRepeatOccs[1][0]+selectedRepeatOccs[0]]) + '\n')
            if maximumRepeatGainValue > 0:
                odd = True
                self.__replaceRepeat(selectedRepeatOccs) #Replacing the chosen repeat
                self.__iterations += 1
        self.__logMessage('---------------')
        self.__logMessage('Number of Iterations: ' + str(self.__iterations))
        self.__createAdjacencyList()
        self.__createDAGGraph()
        self.__nodeStringsGenerate()
    #Returns the cost of the DAG according to the selected costFunction
    def DAGCost(self, costFunction):
        if costFunction == CostFunction.ConcatenationCost:
            return len(self.__concatenatedDAG)-2*len(self.__separatorInts)
        if costFunction == CostFunction.EdgeCost:
            return len(self.__concatenatedDAG)-len(self.__separatorInts)
    #Replaces a repeat's occurrences with a new symbol and creates a new node in the DAG
    def __replaceRepeat(self, input):
        (repeatLength, (repeatOccs)) = input



        repeat = self.__concatenatedDAG[repeatOccs[0]:repeatOccs[0]+repeatLength]
        newTmpConcatenatedDAG = []
        newTmpConcatenatedNTs = []
        prevIndex = 0
        for i in repeatOccs:
            newTmpConcatenatedDAG += self.__concatenatedDAG[prevIndex:i] + [self.__nextNewInt]
            newTmpConcatenatedNTs += self.__concatenatedNTs[prevIndex:i] + [self.__concatenatedNTs[i]]
            prevIndex = i+repeatLength
        self.__concatenatedDAG = newTmpConcatenatedDAG + self.__concatenatedDAG[prevIndex:]
        self.__concatenatedNTs = newTmpConcatenatedNTs + self.__concatenatedNTs[prevIndex:]
        self.__concatenatedDAG = self.__concatenatedDAG + repeat
        self.__concatenatedNTs = self.__concatenatedNTs + [self.__nextNewInt for j in range(repeatLength)]
        self.__logMessage('Added Node: ' +  str(self.__nextNewInt))
        self.__nextNewInt += 2
        self.__concatenatedDAG = self.__concatenatedDAG + [self.__nextNewInt]
        self.__concatenatedNTs = self.__concatenatedNTs + [self.__nextNewInt]
        self.__separatorInts.add(self.__nextNewInt)
        self.__separatorIntsIndices = set([])
        for i in range(len(self.__concatenatedDAG)):
            if self.__concatenatedDAG[i] in self.__separatorInts:
                self.__separatorIntsIndices.add(i)
        self.__nextNewInt += 2
    #Retrieves the maximum-gain repeat (randomizes within ties).
    #Output is a tuple: "(RepeatGain, (RepeatLength, (RepeatOccurrences)))"
    #1st entry of output is the maximum repeat gain value
    #2nd entry of output is a tuple of form: "(selectedRepeatLength, selectedRepeatOccsList)"
    def __retreiveMaximumGainRepeat(self, repeatClass, costFunction):
        repeats = self.__extractRepeats(repeatClass)
        maxRepeatGain = 0
        candidateRepeats = []
        for r in repeats: #Extracting maximum repeat
            repeatStats = r.split()
            repeatOccs = self.__extractNonoverlappingRepeatOccurrences(int(repeatStats[0]),map(int,repeatStats[2][1:-1].split(',')))
            if maxRepeatGain < self.__repeatGain(int(repeatStats[0]), len(repeatOccs), costFunction):
                maxRepeatGain = self.__repeatGain(int(repeatStats[0]), len(repeatOccs), costFunction)
                candidateRepeats = [(int(repeatStats[0]),len(repeatOccs),repeatOccs)]
            else:
                if maxRepeatGain > 0 and maxRepeatGain == self.__repeatGain(int(repeatStats[0]), len(repeatOccs), costFunction):
                    candidateRepeats.append((int(repeatStats[0]),len(repeatOccs),repeatOccs))
        if(len(candidateRepeats) == 0):
            return (-1, (0, []))
        #Randomizing between candidates with maximum gain
        #selectedRepeatStats = candidateRepeats[random.randrange(len(candidateRepeats))]
        selectedRepeatStats = candidateRepeats[0]
        selectedRepeatLength = selectedRepeatStats[0]
        selectedRepeatOccs = sorted(selectedRepeatStats[2])
        return (maxRepeatGain, (selectedRepeatLength, selectedRepeatOccs))
    #Returns the repeat gain, according to the chosen cost function
    def __repeatGain(self, repeatLength, repeatOccsLength, costFunction):
        # if costFunction == CostFunction.ConcatenationCost:
        return (repeatLength-1)*(repeatOccsLength-1)
        # if costFunction == CostFunction.EdgeCost:
        #     return (repeatLength-1)*(repeatOccsLength-1)-1
    #Extracts the designated class of repeats (Assumes ./repeats binary being in the same directory)
    #Output is a string, each line containing: "RepeatLength    NumberOfOccurrence  (CommaSeparatedOccurrenceIndices)"
    def __extractRepeats(self, repeatClass):
        process = subprocess.Popen(["./repeats1/repeats11", "-i", "-r"+repeatClass, "-n2", "-psol"],stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        process.stdin.write(' '.join(map(str,self.__concatenatedDAG)))
        text_file = ''
        while process.poll() is None:
            output = process.communicate()[0].rstrip()
            text_file += output
        process.wait()
        repeats=[]
        firstLine = False
        for line in text_file.splitlines():
            if firstLine == False:
                firstLine = True
                continue
            repeats.append(line.rstrip('\n'))
        return repeats
    #Extracts the non-overlapping occurrences of a repeat from a list of occurrences (scans from left to right)
    def __extractNonoverlappingRepeatOccurrences(self, repeatLength, occurrencesList):
        nonoverlappingIndices = []
        for i in range(len(occurrencesList)):
            if len(nonoverlappingIndices) > 0:
                if (nonoverlappingIndices[-1] + repeatLength <= occurrencesList[i]):#Not already covered
                    nonoverlappingIndices += [occurrencesList[i]]
            else:
                nonoverlappingIndices += [occurrencesList[i]]
        return  nonoverlappingIndices
    #Creates the adjacency list
    def __createAdjacencyList(self):
        separatorPassed = False
        for i in range(len(self.__concatenatedDAG)):
            if i not in self.__separatorIntsIndices:
                node = self.__concatenatedNTs[i]
                if separatorPassed and node == 0:
                    self.__DAG[node].append([])
                    separatorPassed = False
                if node not in self.__DAG:
                    if node == 0:#Target node
                        self.__DAG[node] = [[self.__concatenatedDAG[i]]]
                    else:
                        self.__DAG[node] = [self.__concatenatedDAG[i]]
                else:
                    if node == 0:#Target node
                        self.__DAG[node][-1].append(self.__concatenatedDAG[i])
                    else:
                        self.__DAG[node].append(self.__concatenatedDAG[i])
            else:
                separatorPassed = True
    #Creates the DAG graph object (adjacency list should already be processed)
    def __createDAGGraph(self):
        for node in self.__DAG:
            self.__DAGGraph.add_node(node)
            if node == 0:
                for l in self.__DAG[node]:
                    for n in l:
                        self.__DAGGraph.add_node(n)
                        self.__DAGGraph.add_edge(n, node)
            else:
                for n in self.__DAG[node]:
                    self.__DAGGraph.add_node(n)
                    self.__DAGGraph.add_edge(n, node)
    #Stores the strings corresponding to each DAG node
    def __nodeStringsGenerate(self):
        for node in nx.nodes(self.__DAGGraph):
            if self.__DAGGraph.in_degree(node) == 0:
                # if self.__dic == {}:
                self.__DAGStrings[node] = str(node)
                # else:
                #     self.__DAGStrings[node] = str(self.__dic[node])
            else:
                if node == 0:
                    self.__DAGStrings[node] = []
                else:
                    self.__DAGStrings[node] = ''
        self. __nodeStringsHelper(0)
    # Helper recursive function
    def __nodeStringsHelper(self, n):
        if self.__DAGStrings[n] != [] and self.__DAGStrings[n] != '':
            return
        if n == 0:
            for l in self.__DAG[n]:
                self.__DAGStrings[n].append('')
                for i in range(len(l)):
                    subnode = l[i]
                    self.__nodeStringsHelper(subnode)
                    # if self.__dic == {}:
                    self.__DAGStrings[n][-1] += ' ' + self.__DAGStrings[subnode]
                    # else:
                    #     self.__DAGStrings[n][-1] += self.__DAGStrings[subnode] + ' '
        else:
            for i in range(len(self.__DAG[n])):
                subnode = self.__DAG[n][i]
                self.__nodeStringsHelper(subnode)
                # if self.__dic == {}:
                self.__DAGStrings[n] += ' ' + self.__DAGStrings[subnode]
                # else:
                #     self.__DAGStrings[n] += self.__DAGStrings[subnode] + ' '
    #Returns node's corresponding string
    def __getNodeString(self, n):
        if n == 0:
            result = []
            for l in self.__DAGStrings[n]:
                result.append(' '.join(l.split()))
            return result
        return ' '.join(self.__DAGStrings[n].split())

    # ...........Path-Centrality Functions........
    #Returns a list of strings, corresponding to the nodes removed from DAG, according to greedy core identification algorithm, based on the threshold of edge removal tau
    def greedyCoreID_ByTau(self, tau):
        numberOfUpwardPaths = {}
        numberOfDownwardPaths = {}
        sources = []
        targets = []
        for node in nx.nodes(self.__DAGGraph):
            if self.__DAGGraph.in_degree(node) == 0:
                sources.append(node)
            if self.__DAGGraph.out_degree(node) == 0:
                targets.append(node)
            numberOfUpwardPaths[node] = 0
            numberOfDownwardPaths[node] = 0
        self.__calculateNumberOfUpwardPaths(sources, targets, numberOfUpwardPaths)
        self.__calculateNumberOfDownwardPaths(sources, targets, numberOfDownwardPaths)
        for t in targets:
            numberOfUpwardPaths[t] = 0
        for s in sources:
            numberOfDownwardPaths[s] = 0
        number_of_initial_paths = numberOfDownwardPaths[0]
        number_of_current_paths = numberOfDownwardPaths[0]
        listOfCentralNodes = []
        centralities = self.__calculateCentralities(numberOfUpwardPaths, numberOfDownwardPaths)
        topCentralNodeInfo = max(centralities, key=lambda x:x[1])
        allMaxes = [k for k in centralities if k[1] == topCentralNodeInfo[1]]
        while topCentralNodeInfo[1] > 0 and float(number_of_current_paths)/float(number_of_initial_paths) > 1-tau:#Node with positive centrality exists
            for nodeToBeRemoved in allMaxes:
                nodeToBeRemoved = nodeToBeRemoved[0]
                self.__DAGGraph.remove_node(nodeToBeRemoved)
                listOfCentralNodes.append(nodeToBeRemoved)
            numberOfUpwardPaths = {}
            numberOfDownwardPaths = {}
            for node in nx.nodes(self.__DAGGraph):
                numberOfUpwardPaths[node] = 0
                numberOfDownwardPaths[node] = 0
            self.__calculateNumberOfUpwardPaths(sources, targets, numberOfUpwardPaths)
            self.__calculateNumberOfDownwardPaths(sources, targets, numberOfDownwardPaths)
            for t in targets:
                numberOfUpwardPaths[t] = 0
            for s in sources:
                numberOfDownwardPaths[s] = 0
            centralities = self.__calculateCentralities(numberOfUpwardPaths, numberOfDownwardPaths)
            topCentralNodeInfo = max(centralities, key=lambda x: x[1])
            allMaxes = [k for k in centralities if k[1] == topCentralNodeInfo[1]]
            number_of_current_paths = numberOfDownwardPaths[0]
        self.__DAGGraph = nx.MultiGraph()
        self.__createDAGGraph()#Reconstructing the DAG graph
        core = []
        for i in range(len(listOfCentralNodes)):
            core.append(self.__getNodeString(listOfCentralNodes[i]))
        return core
    # Returns a list of strings, corresponding to the nodes removed from DAG, according to greedy core identification algorithm, based on the cardinality of the extracted set
    def greedyCoreID_ByCardinality(self, k):
        numberOfUpwardPaths = {}
        numberOfDownwardPaths = {}
        sources = []
        targets = []
        for node in nx.nodes(self.__DAGGraph):
            if self.__DAGGraph.in_degree(node) == 0:
                sources.append(node)
            if self.__DAGGraph.out_degree(node) == 0:
                targets.append(node)
            numberOfUpwardPaths[node] = 0
            numberOfDownwardPaths[node] = 0
        self.__calculateNumberOfUpwardPaths(sources, targets, numberOfUpwardPaths)
        self.__calculateNumberOfDownwardPaths(sources, targets, numberOfDownwardPaths)
        for t in targets:
            numberOfUpwardPaths[t] = 0
        for s in sources:
            numberOfDownwardPaths[s] = 0
        number_of_initial_paths = numberOfDownwardPaths[0]
        number_of_current_paths = numberOfDownwardPaths[0]
        listOfCentralNodes = []
        centralities = self.__calculateCentralities(numberOfUpwardPaths, numberOfDownwardPaths)
        topCentralNodeInfo = max(centralities, key=lambda x: x[1])
        allMaxes = [k for k in centralities if k[1] == topCentralNodeInfo[1]]
        while topCentralNodeInfo[1] > 0 and len(listOfCentralNodes) <= k:  # Node with positive centrality exists
            for nodeToBeRemoved in allMaxes:
                nodeToBeRemoved = nodeToBeRemoved[0]
                self.__DAGGraph.remove_node(nodeToBeRemoved)
                listOfCentralNodes.append(nodeToBeRemoved)
            numberOfUpwardPaths = {}
            numberOfDownwardPaths = {}
            for node in nx.nodes(self.__DAGGraph):
                numberOfUpwardPaths[node] = 0
                numberOfDownwardPaths[node] = 0
            self.__calculateNumberOfUpwardPaths(sources, targets, numberOfUpwardPaths)
            self.__calculateNumberOfDownwardPaths(sources, targets, numberOfDownwardPaths)
            for t in targets:
                numberOfUpwardPaths[t] = 0
            for s in sources:
                numberOfDownwardPaths[s] = 0
            centralities = self.__calculateCentralities(numberOfUpwardPaths, numberOfDownwardPaths)
            topCentralNodeInfo = max(centralities, key=lambda x: x[1])
            allMaxes = [k for k in centralities if k[1] == topCentralNodeInfo[1]]
            number_of_current_paths = numberOfDownwardPaths[0]
        self.__DAGGraph = nx.MultiGraph()
        self.__createDAGGraph()  # Reconstructing the DAG graph
        core = []
        for i in range(len(listOfCentralNodes)):
            core.append(self.__getNodeString(listOfCentralNodes[i]))
        return core
    #Calculates the centralities for all nodes
    def __calculateCentralities(self, numberOfUpwardPaths, numberOfDownwardPaths):
        result = []
        for node in nx.nodes(self.__DAGGraph):
            result.append((node, numberOfUpwardPaths[node] * numberOfDownwardPaths[node]))
        return result
    #Calculates the number of Upward paths for all nodes
    def __calculateNumberOfUpwardPaths(self, sources, targets, numberOfUpwardPaths):
        for n in sources:
            self.__dfsUpward(n, sources, targets, numberOfUpwardPaths)
    # Helper recursive function
    def __dfsUpward(self, n, sources, targets, numberOfUpwardPaths):
        if self.__DAGGraph.out_degree(n) == 0:
            numberOfUpwardPaths[n] = 1
            return
        elif numberOfUpwardPaths[n] > 0:
            return
        else:
            for o in self.__DAGGraph.out_edges(n):
                self.__dfsUpward(o[1], sources, targets, numberOfUpwardPaths)
                numberOfUpwardPaths[n] += numberOfUpwardPaths[o[1]]
    # Calculates the number of Downward paths for all nodes
    def __calculateNumberOfDownwardPaths(self, sources, targets, numberOfDownwardPaths):
        for n in targets:
            self.__dfsDownward(n, sources, targets, numberOfDownwardPaths)
    # Helper recursive function
    def __dfsDownward(self, n, sources, targets, numberOfDownwardPaths):
        if self.__DAGGraph.in_degree(n) == 0:
            numberOfDownwardPaths[n] = 1
            return
        elif numberOfDownwardPaths[n] > 0:
            return
        else:
            for o in self.__DAGGraph.in_edges(n):
                self.__dfsDownward(o[0], sources, targets, numberOfDownwardPaths)
                numberOfDownwardPaths[n] += numberOfDownwardPaths[o[0]]

    # ...........Printing Functions........
    # Prints the DAG, optionally in integer form if intDAGPrint==True
    def printDAG(self, intDAGPrint):
        self.__logMessage('DAGCost(Concats): ' + str(self.DAGCost(CostFunction.ConcatenationCost)))
        self.__logMessage('DAGCost(Edges):' + str(self.DAGCost(CostFunction.EdgeCost)))
        DAG = self.__concatenatedDAG
        # print 'dag'
        # print DAG
        NTs = self.__concatenatedNTs
        # print 'nts'
        # print NTs
        separatorInts = self.__separatorInts
        Dic = self.__dic
        nodes = {}
        ntDic = {}
        counter = 1
        NTsSorted = set([])
        for i in range(len(NTs)):
            if NTs[i] not in ntDic and NTs[i] not in separatorInts:
                NTsSorted.add(NTs[i])
                # ntDic[NTs[i]] = 'N'+str(counter)
                # nodes['N'+str(counter)] = ''
                ntDic[NTs[i]] = 'N' + str(NTs[i])
                nodes['N' + str(NTs[i])] = ''
                counter += 1
        for i in range(len(DAG)):
            if DAG[i] not in NTsSorted:
                if DAG[i] not in separatorInts:
                    if not intDAGPrint:
                        try:
                            nodes[ntDic[NTs[i]]] = str(nodes[ntDic[NTs[i]]]) + ' ' + str(Dic[DAG[i]])
                        except:
                            print (DAG[i], NTs[i])
                            raise
                    else:
                        nodes[ntDic[NTs[i]]] = str(nodes[ntDic[NTs[i]]]) + ' ' + str(DAG[i])
                else:
                    nodes[ntDic[NTs[i - 1]]] = str(nodes[ntDic[NTs[i - 1]]]) + ' ||'
            else:
                if not intDAGPrint:
                    try:
                        nodes[ntDic[NTs[i]]] = str(nodes[ntDic[NTs[i]]]) + ' ' + str(ntDic[DAG[i]])
                    except:
                        print (DAG[i], NTs[i])
                        raise
                else:
                    nodes[ntDic[NTs[i]]] = str(nodes[ntDic[NTs[i]]]) + ' ' + str(ntDic[DAG[i]])
        NTsSorted = sorted(list(NTsSorted))
        nodeCounter = 0
        for nt in NTsSorted:
            if intDAGPrint:
                subnodes = nodes[ntDic[nt]].rstrip(' ||').split(' ||')
                for s in subnodes:
                    print (ntDic[nt] + ' ->' + s)
            else:
                subnodes = nodes[ntDic[nt]].rstrip(' ||').split(' ||')
                for s in subnodes:
                    print(ntDic[nt] + ' -> ' + s)
            nodeCounter += 1
    # Log via flags
    def __logViaFlag(self, flag):
        if not self.__quietLog:
            if flag == LogFlag.ConcatenationCostLog:
                sys.stderr.write('DAGCost(Concats): ' + str(self.DAGCost(CostFunction.ConcatenationCost)) + '\n')
                print(str('DAGCost(Concats): ' + str(self.DAGCost(CostFunction.ConcatenationCost))))
            if flag == LogFlag.EdgeCostLog:
                sys.stderr.write('DAGCost(Edges): ' + str(self.DAGCost(CostFunction.EdgeCost)) + '\n')
                print(str('DAGCost(Edges): ' + str(self.DAGCost(CostFunction.EdgeCost))))
    # Log custom message
    def __logMessage(self, message):
        if not self.__quietLog:
            sys.stderr.write(message + '\n')
            print(str(message))

    # ...........Utility Functions........
    # Converts the input data into an integer sequence, returns the integer sequence and the dictionary for recovering orginal letters
    def __preprocessInput(self, inputFile, charSeq=SequenceType.Character, noNewLineFlag=True):
        if charSeq == SequenceType.Character:  # Building an integer-spaced sequence from the input string
            letterDict = {}
            counterDict = {}
            i = 0
            counter = 1
            newContents = ''
            if noNewLineFlag:
                line = inputFile.read()
                for i in range(len(line)):
                    if line[i] not in counterDict:
                        letterDict[counter] = line[i]
                        counterDict[line[i]] = counter
                        counter += 1
                    newContents += str(counterDict[line[i]]) + ' '
            else:
                for line in inputFile:
                    line = line.rstrip('\n')
                    for i in range(len(line)):
                        if line[i] not in counterDict:
                            letterDict[counter] = line[i]
                            counterDict[line[i]] = counter
                            counter += 1
                        newContents += str(counterDict[line[i]]) + ' '
                    newContents += '\n'
            return (newContents.rstrip('\n'), letterDict)
        if charSeq == SequenceType.Integer:  # input is space seperated integers
            newContents = ''
            dict = {}
            for l in inputFile.read().splitlines():
                line = l.split()
                for i in range(len(line)):
                    if not isinstance(int(line[i]), int) or line[i] == ' ':
                        raise ValueError('Input file is not in space-separated integer form.')
                    else:
                        dict[int(line[i])] = line[i]
                newContents += l + '\n'
            return (newContents.rstrip('\n'), dict)
        if charSeq == SequenceType.SpaceSeparated:  # input is space-seperated words
            wordDict = {}
            counterDict = {}
            i = 0
            counter = 1
            newContents = ''
            for line in inputFile:
                line = line.rstrip('\n')
                for w in line.split():
                    if w not in counterDict:
                        wordDict[counter] = w
                        counterDict[w] = counter
                        counter += 1
                    newContents += str(counterDict[w]) + ' '
                newContents += '\n'
            return (newContents.rstrip('\n'), wordDict)

#Sets the value of parameters
def processParams(argv):
    chFlag = SequenceType.Character #if false, accepts integer sequence
    printIntsDAG = False #if true, prints the DAG in integer sequence format
    quietLog = False #if true, disables logging
    rFlag = 'mr' #repeat type (for normal repeat replacements)
    functionFlag = 'e' #cost function to be optimized
    noNewLineFlag = True #consider each line as a separate string
    loadDAGFlag = False

    usage = """Usage: ./python Lexis.py [-t (c | i | s) | -p (i) | -q | -r (r | mr | lmr | smr) | -f (c | e) | -m | -l] <filename>
    [-t]: choosing between character sequence, integer sequence or space-separated sequence
        c - character sequence
        i - integer sequence
        s - space-separated sequence
    [-p]: specifies DAG printing option (for debugging purposes)
        i - prints the DAG in integer sequence format
    [-q]: disables logging
    [-r]: repeat type (for normal repeat replacements)
        r - repeat
        mr - maximal repeat (default)
        lmr - largest-maximal repeat
        smr - super-maximal repeat
    [-f]: cost function to be optimized
        c - concatenation cost
        e - edge cost (default)
    [-m]: consider each line of the input file as a separate target string
    [-l]: load a DAG file (will override -r -t -m options)
                    """
    if len(argv) == 1 or (len(argv) == 2 and argv[1] == '-h'):
        sys.stderr.write('Invalid input\n')
        sys.stderr.write(usage + '\n')
        sys.exit()
    optlist,args = getopt.getopt(argv[1:], 't:p:qr:f:ml')
    for opt,arg in optlist:
        if opt == '-t':
            for ch in arg:
                if ch == 'c' or ch == 'i' or ch == 's':
                    chFlag = ch
                else:
                    sys.stderr.write('Invalid input in ' + '-i' + ' flag\n')
                    sys.stderr.write(usage + '\n')
                    sys.exit()
        if opt == '-p':
            for ch in arg:
                if ch == 'i':
                    printIntsDAG = True
                else:
                    sys.stderr.write('Invalid input in ' + '-p' + ' flag\n')
                    sys.stderr.write(usage + '\n')
                    sys.exit()
        if opt == '-q':
            quietLog = True
        if opt == '-r':
            if arg == 'r' or arg == 'mr' or arg == 'lmr' or arg == 'smr':
                rFlag = arg
            else:
                sys.stderr.write('Invalid input in ' + '-r' + ' flag\n')
                sys.stderr.write(usage + '\n')
                sys.exit()
        if opt == '-f':
            if arg == 'c' or arg == 'e':
                functionFlag = arg
            else:
                sys.stderr.write('Invalid input in ' + '-f' + ' flag\n')
                sys.stderr.write(usage + '\n')
                sys.exit()
        if opt == '-m':
            noNewLineFlag = False
        if opt == '-l':
            loadDAGFlag = True
    return (chFlag, printIntsDAG, quietLog, rFlag, functionFlag, noNewLineFlag, loadDAGFlag)

if __name__ == "__main__":
    (chFlag, printIntsDAG, quietLog, rFlag, functionFlag, noNewLineFlag, loadDAGFlag) = processParams(sys.argv)
    g = DAG(open(sys.argv[-1],'r'), loadDAGFlag, chFlag, noNewLineFlag)
    g.GLexis(quietLog, rFlag, functionFlag)
    g.printDAG(printIntsDAG)

    #If desired to see the central nodes, please uncomment the lines below
    # centralNodes = g.greedyCoreID_ByTau(0.95)
    # print
    # print 'Central Nodes:'
    # for i in range(len(centralNodes)):
    #     print centralNodes[i]