# -*- coding: utf-8 -*-
"""
Author:
    Hector Quijada

Fuzzy ARTMAP implemetation using ARTa, ARTb, and Map Field

Nodes at ARTa and ARTb increase and dimensions at Map Field are adjusted accordingly

G. A. Carpenter, S. Grossberg, N. Markuzon, J. H. Reynolds and D. B. Rosen, "Fuzzy ARTMAP: A neural network architecture 
for incremental supervised learning of analog multidimensional maps," in IEEE Transactions on Neural Networks, vol. 3, 
no. 5, pp. 698-713, Sept. 1992, doi: 10.1109/72.159059. keywords: {Fuzzy neural networks;Neural networks;Fuzzy logic;
Resonance;Subspace constraints;Computational modeling;Fuzzy systems;Supervised learning;Multidimensional systems;Fuzzy sets},


"""
#Import libraries

import numpy as np
import time
import os
import errno

class FuzzyArtMap:
    
    def __init__(self, ARTa_nodes = 1, ARTb_nodes = 1, baseline_vigilance=0.9):
        
        #Initial number of nodes
        self.ARTa_nodes = ARTa_nodes
        self.ARTb_nodes = ARTb_nodes
        
        #Vigilance Parameters
        self.baseline_vigilance = baseline_vigilance
        self.rho_a = self.baseline_vigilance
        self.rho_b = 0.9
        self.rho_ab = 0.95
        
        #Learning Rates
        self.beta_a = 1.0
        self.beta_b = 1.0
        self.beta_ab = 1.0
        
        #Commited Learning Rates
        self.committed_beta_a = 0.75
        self.committed_beta_b = 0.75
        self.committed_beta_ab = 0.75
        
        #Choice Parameter
        self.alpha = 0.001
        
        #Commited nodes
        self.committed_nodes = set()
        
        #Reset nodes
        self.reset_nodes = []
        
        #Resonance Flag
        self.resonance = False
        
        #Match Tracking increase
        self.epsilon = 0.001
        
        #Epochs Variable
        self.epochs = 0
        
        #Weights Variable for prediction
        self.weight_a = 0
        self.weight_b = 0
        self.weight_ab = 0
        
        #Obtain path to save files
        self._path = self._ART_path()
    
    #Define relative path to weights
    def _ART_path(self):
        
        # Get the directory of the current script
        dirname = os.path.dirname(os.path.abspath(__file__))
        
        # Define the relative path components
        relative_path_components = ['train_files','ART_weight_files']

        # Join the directory path and the relative path components
        folder_name = os.path.join(dirname, *relative_path_components)
        
        #Create folder if does not exist
        try:
            os.mkdir(folder_name)
            print("Directory %s created.", folder_name)
            
        except OSError as e:
            
            if e.errno == errno.EEXIST:
                print('ART train files directory %s already exists; not created.', folder_name)

        return folder_name
    
    #Extract inputs and outputs relation in a csv file, give dimensions for each input for training purposes
    def _extract_csv_inputs(self, Ia_dim, Ib_dim, path):
        
        #Initialize Auxiliary Inputs
        Ia_aux = np.zeros(Ia_dim)
        Ib_aux = np.zeros(Ib_dim)
        
        #Initialize inputs as numpy array
        Ia = np.zeros(Ia_dim)
        Ib = np.zeros(Ib_dim)
        
        #Extract all Inputs from train file as a single numpy array for ARTa and ARTb
        with open(path, 'r') as file:
            
            # Read all lines and process each one
            lines = file.readlines()
            for line in lines:
                inputs = line.strip().split(',') #Extract EOL and split in a list with separator ","
                
                #Split into two vectors and convert to float
                for i in range(Ia_dim):
                    Ia_aux[i] = float(inputs[i]) 
                
                for i in range(Ib_dim):
                    Ib_aux[i] = float(inputs[Ia_dim+i])
                
                #Stack current line vector into a matrix
                Ia = np.vstack((Ia, Ia_aux))
                Ib = np.vstack((Ib, Ib_aux))
            
            #Delete first row as it was auxiliary
            Ia = np.delete(Ia, 0, axis=0)
            Ib = np.delete(Ib, 0, axis=0)
        
        return Ia, Ib
    
    #Extract weights of a generated training csv file, for prediction purposes
    def _extract_csv_weight(self, weights_path):

        #Extract all Inputs from train file as a single numpy array for ARTa and ARTb
        with open(weights_path, 'r') as file:
            
            # Read all lines and process each one
            lines = file.readlines()
            
            #Obtain number columns as it defines vector size
            w_dim = lines[0].count(',') + 1 
            
            w = np.zeros(w_dim)
            w_aux = np.zeros(w_dim)
            
            for line in lines:

                w_str = line.strip().split(',') #Extract EOL and split in a list with separator ","
                
                #Split into two vectors and convert to floar
                for i in range(len(w_str)):
                    w_aux[i] = float(w_str[i])
                
                #Stack current line vector into a matrix
                w = np.vstack((w, w_aux))
                #Delete first row as it was auxiliary
            
            w = np.delete(w, 0, axis=0)
        
        return w
    
    #Save weights results from training
    def _save_csv_weights(self, weight_a, weight_b, weight_ab):
        
        #Set weights file path
        weights_a_path = os.path.join(self._path, "weights_a.csv")
        weights_b_path = os.path.join(self._path, "weights_b.csv")
        weights_ab_path = os.path.join(self._path, "weights_ab.csv")
        
        #Save weights file
        np.savetxt(weights_a_path, weight_a, delimiter=',', fmt='%f')
        np.savetxt(weights_b_path, weight_b, delimiter=',', fmt='%f')
        np.savetxt(weights_ab_path, weight_ab, delimiter=',', fmt='%f')

        print("Weights updated!")
    
    #Add complement to input
    def _complement_encode(self,Ia,Ib):
        
        complement_encoded_Ia = np.zeros(Ia.shape[1]*2)
        complement_encoded_Ib = np.zeros(Ib.shape[1]*2)
        
        #Rearrange Inputs with complements
        for i in range(Ia.shape[0]):
            complement_encoded_Ia = np.vstack((complement_encoded_Ia,np.concatenate((Ia[i], 1-Ia[i]))))
            complement_encoded_Ib = np.vstack((complement_encoded_Ib,np.concatenate((Ib[i], 1-Ib[i]))))
                
        complement_encoded_Ia = np.delete(complement_encoded_Ia, 0, axis=0)
        complement_encoded_Ib = np.delete(complement_encoded_Ib, 0, axis=0)
        
        return complement_encoded_Ia, complement_encoded_Ib
    
    #Search for input resonance given a set of nodes
    def _resonance_search(self,weight,Ii,rho, reset_nodes=[]):
        
        #Reset nodes
        self.reset_nodes = reset_nodes
        
        #Resonance Flag
        self.resonance = False
        
        #Initialize T
        T = np.zeros((weight.shape[0]))

        #Obtain activation value for each node at ARTa
        for j in range(weight.shape[0]):
            category_choice_numerator = np.sum(np.minimum(Ii, weight[j]))
            category_choice_denominator = self.alpha + np.sum(weight[j])
            category_choice = category_choice_numerator / category_choice_denominator
            T[j] = category_choice
        
        while not self.resonance:
            
            #If node already reset set activation value to 0 to avoid it competing
            if len(self.reset_nodes) > 0:
                for reset in self.reset_nodes:
                    T[reset] = 0
            
            #Obtain max activation value
            J = np.argmax(T) #If more than 1 max value first index is implicitely selected
            
            #Compute current vigilance value to compare with criteria rho_a
            vigilance_criteria_numerator = np.sum(np.minimum(Ii, weight[J]))
            vigilance_criteria_denominator = np.sum(Ii)
            vigilance_criteria = vigilance_criteria_numerator/vigilance_criteria_denominator
            
            if vigilance_criteria >= rho:
                self.resonance = True
            else:
                if len(self.reset_nodes) < len(T):
                    self.reset_nodes.append(J)
                else:
                    #We break while loop and resonance in Fuzzy ART remains false to trigger learning of new
                    #created node
                    
                    #Add new node to learn new pattern
                    weight = np.vstack((weight, np.ones((1,Ii.shape[0]))))
                    
                    J = len(T)
                    break
        
        return J, self.resonance, vigilance_criteria, weight
    
    #Train Fuzzy ARTMAP network with a set of input, outputs relations
    def train(self, Ia_dim, Ib_dim, train_path, save_weights=True, load_csv=True, Ia_retrain = [], Ib_retrain = []):
        
        if load_csv == True:
            #Extract inputs from csv 
            Ia, Ib = self._extract_csv_inputs(Ia_dim, Ib_dim, train_path)
            
            #Add complement encoding to original inputs
            Iac, Ibc = self._complement_encode(Ia, Ib)
            
            #Initialize all weights to 1
            weight_a = np.ones((self.ARTa_nodes, Ia_dim*2))
            weight_b = np.ones((self.ARTb_nodes, Ib_dim*2))
            weight_ab = np.ones((self.ARTa_nodes, self.ARTb_nodes))
        else:
            weight_a = self.weight_a
            weight_b = self.weight_b
            weight_ab = self.weight_ab
            
            Ia = Ia_retrain
            Ib = Ib_retrain
            
            #Add complement encoding to original inputs
            Iac, Ibc = self._complement_encode(Ia, Ib)

        #Loop Training for each set of inputs Ia, Ib (Must be same number of inputs)
        for i in range(Ia.shape[0]):

            #ART a nodes reset by Map Field
            reset_nodes_map = []

            #Flag for pattern disprove when entering
            pattern_disprove = False

            #Resonance Flags(ARTa, ARTb, Map Field)
            resonance_a = False
            resonance_b = False
            resonance_ab = False

            #ARTa Resonance Search with baseline vigilance
            Ja, resonance_a, rho_a, weight_a = self._resonance_search(weight=weight_a, Ii=Iac[i], rho=self.rho_a, reset_nodes=[])
            #ARTb Resonance Search
            Jb, resonance_b, rho_b, weight_b = self._resonance_search(weight=weight_b, Ii=Ibc[i], rho=self.rho_b, reset_nodes=[])

            if resonance_a and not resonance_b:
                weight_ab = np.hstack((weight_ab, np.zeros((self.ARTa_nodes,1))))
                self.ARTb_nodes += 1
                weight_ab = np.vstack((weight_ab, np.ones((1,self.ARTb_nodes))))
                self.ARTa_nodes += 1

            else:
                #Update weights at mapfield
                if not resonance_b:
                    weight_ab = np.hstack((weight_ab, np.zeros((self.ARTa_nodes,1))))
                    self.ARTb_nodes += 1
                if not resonance_a:
                    weight_ab = np.vstack((weight_ab, np.ones((1,self.ARTb_nodes))))
                    self.ARTa_nodes += 1

            if not resonance_a and not resonance_b:
                pattern_disprove = True

            #Match Tracking at map field Fab
            while not resonance_ab:
                
                #Initialize y_b node
                y_b = np.zeros(self.ARTb_nodes)
                y_b[Jb] = 1
                
                #Evaluate if the nodes at F2a and F2b have been activated
                if resonance_a == True and resonance_b == True:
                    x_ab = np.minimum(y_b, weight_ab[Ja]) 
                elif resonance_a == True and resonance_b == False:
                    x_ab = np.sum(weight_ab[Ja])
                    
                    #ARTb output disregarded ARTa output, that is because we did not found the output in current
                    #weights at ARTb, but we did at ARTa, that is a contradiction, since we are doing a relation
                    #between inputs and outputs, that is as saying I found something that does not exist yet
                    #that is why you need to consider the node at ARTb as the parent for this weight update and also
                    #add a new neuron at ARTa, since we know already the prediction is not correct
                    Ja = Jb
                    
                    #Do not add weight at last input
                    #if i != (Ia.shape[0]-1) and pattern_disprove == False:
                    if pattern_disprove == False:
                        weight_a = np.vstack((weight_a, np.ones((1,Ia_dim*2))))
                        
                elif resonance_a == False and resonance_b == True:
                    x_ab = y_b
                elif resonance_a == False and resonance_b == False:
                    x_ab = 0.0
                    
                #Compute current vigilance value to compare with criteria rho_ab
                map_vigilance_criteria = np.sum(x_ab)/np.sum(y_b)
                
                if map_vigilance_criteria >= self.rho_ab:
                    
                    #Update learning rate if nodes already commited
                    
                    beta_b = self.beta_b

                    if Ja in self.committed_nodes:
                        beta_a = self.committed_beta_a
                        beta_ab = self.committed_beta_ab
                    else:
                        beta_a = self.beta_a
                        beta_ab = self.beta_ab
                    
                    #Update weights based on learning law
                    weight_a[Ja] = (beta_a * (np.minimum(Iac[i],weight_a[Ja]))) + ((1 - beta_a) * weight_a[Ja])
                    weight_b[Jb] = (beta_b * (np.minimum(Ibc[i],weight_b[Jb]))) + ((1 - beta_b) * weight_b[Jb])
                    weight_ab[Ja] = (beta_ab * (np.minimum(y_b,weight_ab[Ja]))) + ((1 - beta_ab) * weight_ab[Ja])
                    self.committed_nodes.add(Ja)

                    resonance_ab = True

                else:
                    #If map field vigilance parameter condition not met increase rho_a
                    
                    rho_a += self.epsilon
                    
                    Ja, resonance_a, rho_a, weight_a = self._resonance_search(weight=weight_a, Ii=Iac[i], rho=rho_a, reset_nodes=reset_nodes_map)
		    
                    if not resonance_a:
                        reset_nodes_map.append(Ja)
                        weight_ab = np.vstack((weight_ab, np.ones((1,self.ARTb_nodes))))
                        self.ARTa_nodes += 1
            
        #Save weights
        if save_weights:
            self._save_csv_weights(weight_a, weight_b, weight_ab)
    
    #Load trained weights once in a prediction loop
    def load_weights(self):
        
        #Set weights file path
        weights_a_path = os.path.join(self._path, "weights_a.csv")
        weights_b_path = os.path.join(self._path, "weights_b.csv")
        weights_ab_path = os.path.join(self._path, "weights_ab.csv")

        self.weight_a = self._extract_csv_weight(weights_path=weights_a_path)
        self.ARTa_nodes = self.weight_a.shape[0]
        self.weight_b = self._extract_csv_weight(weights_path=weights_b_path)
        self.ARTb_nodes = self.weight_b.shape[0]
        self.weight_ab = self._extract_csv_weight(weights_path=weights_ab_path)
    
    #For testing purposes, create a csv file and extract inputs
    def extract_csv_input(self, Ia_dim, path):
        
        #Initialize Auxiliary Inputs
        Ia_aux = np.zeros(Ia_dim)
        
        #Initialize inputs as numpy array
        Ia = np.zeros(Ia_dim)
        
        #Extract all Inputs from train file as a single numpy array for ARTa and ARTb
        with open(path, 'r') as file:
            
            # Read all lines and process each one
            lines = file.readlines()
            for line in lines:
                inputs = line.strip().split(',') #Extract EOL and split in a list with separator ","
                
                #Split into two vectors and convert to float
                for i in range(Ia_dim):
                    Ia_aux[i] = float(inputs[i]) 
                
                #Stack current line vector into a matrix
                Ia = np.vstack((Ia, Ia_aux))
            
            #Delete first row as it was auxiliary
            Ia = np.delete(Ia, 0, axis=0)
        
        complement_encoded_Ia = np.zeros(Ia.shape[1]*2)
        
        #Rearrange Inputs with complements
        for i in range(Ia.shape[0]):
            complement_encoded_Ia = np.vstack((complement_encoded_Ia,np.concatenate((Ia[i], 1-Ia[i]))))
                
        complement_encoded_Ia = np.delete(complement_encoded_Ia, 0, axis=0)
        
        return complement_encoded_Ia
    
    #Predict given a set of trained weights
    def predict(self,Ia,rho_a):
    	
        self.rho_a = rho_a
        
    	#Resonance Flags(ARTa, ARTb, Map Field)
        resonance_a = False
            
        #ARTa Resonance Search with baseline vigilance
        Ja, resonance_a, rho_a, self.weight_a = self._resonance_search(weight=self.weight_a, Ii=Ia, rho=self.rho_a, reset_nodes=[])
        
        #If category found among existing ones return Jab prediction
        #else return none as it means pattern is not recognized by network and needs to be classified
        if resonance_a:
            
            #Category is labeled as the the weight not zer amongst the columns of weights at map field from F2a
            ja_wab = self.weight_ab[Ja]
            Jab = np.nonzero(ja_wab)[0].item()
            return [Jab, Ia]
    
        else:
            
            #Update ARTa neuron number and weight ab size
            self.ARTa_nodes += 1
            self.weight_ab = np.vstack((self.weight_ab, np.ones((1,self.ARTb_nodes))))
            
            return [None, Ia]
        
    #If when doing predictions training is canceled remove unnecesary weights
    def remove_prediction_weight(self):
        #Remove new node since training was canceled
        self.weight_a = np.delete(self.weight_a, -1, axis=0)
        self.ARTa_nodes -= 1

        #Remove new node at field map ab
        self.weight_ab = np.delete(self.weight_ab, -1, axis=0)

        
