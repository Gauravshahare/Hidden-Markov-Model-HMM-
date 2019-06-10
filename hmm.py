### .............................................................Implementation of hidden markov model.................................................................
## 3 Taks to train HMM
  ## 1) Forward Algorithm
  ## 2) Backward Algorithm
  ## 3) Forward Backward Algorithm

import numpy as np
import pandas as pd

class HMM:
    def __init__(self,sequence,transition_matrix,emission_matrix,initial_matrix,number_states=5,number_output=5,n_epochs=25):
        self.seq=sequence
        self.tras=transition_matrix
        self.emiss=emission_matrix
        self.iprob=initial_matrix
        self.states=number_states
        self.out=number_output
        self.epochs=n_epochs
        
    def forward(self):                                                            ## returns the probability of the sequence
        nstates=self.states                                                       ## number of states
        nlength=len(self.seq)                                                     ## length of sequnce
        dp=np.zeros(shape=(nstates,nlength))                                      ## intialize 2D np array for dynamic algorithm
        
        for j in range(nlength):
            for i in range(nstates):                         
                if j==0:
                    dp[i][j]=self.iprob[i]*self.emiss[i][self.seq[j]]
                else:
                    for k in range(nstates):
                        dp[i][j]+=(dp[k][j-1]*self.tras[k][i]*self.emiss[i][self.seq[j]])
                        
        ## Time complexity O(N*N*T) Where N=nstates  and  T=nlength
        
        tp=0.0
        for i in range(nstates):
            tp+=dp[i][nlength-1]
        ## Adding up the Probabilty in the last step           
        return tp
    
    def decoding(self):                                                           ## return the list of most probable hidden states
        nstates=self.states                                                       ## number of states
        nlength=len(self.seq)                                                     ## length of sequnce
        dp=np.zeros(shape=(nstates,nlength))
        maxi=0
        index=0
        
        for j in range(nlength):
            for i in range(nstates):                         
                if j==0:
                    dp[i][j]=self.iprob[i]*self.emiss[i][self.seq[j]]
                else:
                    for k in range(nstates):
                        dp[i][j]+=(dp[k][j-1]*self.tras[k][i]*self.emiss[i][self.seq[j]])
                if j==nlength-1:
                    if dp[i][j]>maxi:
                        maxi=dp[i][j]
                        index=i
        
        hstates=np.zeros(shape=(1,nlength))
        for j in range(nlength-1,-1,-1):
            hstates[j]=index
            if(j==0):
                break
            for i in range(nstates):
                dp[index][j]==dp[i][j-1]*self.tras[i][index]*self.emiss[i][self.seq[j-1]]
                index=i
                break
        return hstates
        
    
    def backward(self,step,state):
        nstates=self.states                                                       ## number of states
        nlength=len(self.seq)                                                     ## length of sequnce
        dp=np.zeros(shape=(nstates,nlength))
        
        for j in range(nlength,-1,-1):
            for i in range(nstates):
                if(j==nlength-1):
                    dp[i][j]=1.0
                else:
                    for k in range(nstates):
                        dp[i][j]+=(self.tras[i][k]*self.emiss[k][self.seq[j+1]]*dp[k][j+1])
        return dp[state][step]
    
    def forward_backward(self):
        
        for epochs in range(self.epochs):
            print("ok")