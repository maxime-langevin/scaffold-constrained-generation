import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_structs import tokenize
from utils import Variable
import re
import random
import numpy as np
from scipy.stats import uniform


def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)

    return string

def tokenize_custom(smiles):
    """Takes a SMILES string and returns a list of tokens.
    This will swap 'Cl' and 'Br' to 'L' and 'R' and treat
    '[xx]' as one token."""
    regex = '(\[[^\[\]]{1,6}\])'
    smiles = replace_halogen(smiles)
    char_list = re.split(regex, smiles)
    tokenized = []
    for char in char_list:
        if char == '*':
            tokenized.append(char)
        if char.startswith('['):
            tokenized.append(char)
        else:
            chars = [unit for unit in char]
            [tokenized.append(unit) for unit in chars]
    tokenized.append('EOS')
    return tokenized

class MultiGRU(nn.Module):
    """ Implements a three layer GRU cell including an embedding layer
       and an output linear layer back to the size of the vocabulary"""
    def __init__(self, voc_size):
        super(MultiGRU, self).__init__()
        self.embedding = nn.Embedding(voc_size, 128)
        self.gru_1 = nn.GRUCell(128, 512)
        self.gru_2 = nn.GRUCell(512, 512)
        self.gru_3 = nn.GRUCell(512, 512)
        self.linear = nn.Linear(512, voc_size)

    def forward(self, x, h):
        x = self.embedding(x)
        h_out = Variable(torch.zeros(h.size()))
        x = h_out[0] = self.gru_1(x, h[0])
        x = h_out[1] = self.gru_2(x, h[1])
        x = h_out[2] = self.gru_3(x, h[2])
        x = self.linear(x)
        return x, h_out

    def init_h(self, batch_size):
        # Initial cell state is zero
        return Variable(torch.zeros(3, batch_size, 512))

class scaffold_constrained_RNN():
    """Implements the Prior and Agent RNN. Needs a Vocabulary instance in
    order to determine size of the vocabulary and index of the END token"""
    def __init__(self, voc):
        self.rnn = MultiGRU(voc.vocab_size)
        if torch.cuda.is_available():
            self.rnn.cuda()
        self.voc = voc
        
    def set_pattern(self, pattern='CCC(*)CC'):
        self.pattern = pattern

    def likelihood(self, target, max_length=140):
        """
            Retrieves the likelihood of a given sequence

            Args:
                target: (batch_size * sequence_lenght) A batch of sequences

            Outputs:
                log_probs : (batch_size) Log likelihood for each example*
                entropy: (batch_size) The entropies for the sequences. Not
                                      currently used.
        """
        batch_size, seq_length = target.size()
        start_token = Variable(torch.zeros(batch_size, 1).long())
        
        start_token[:] = self.voc.vocab['GO']
        x = torch.cat((start_token, target[:, :-1]), 1)
        h = self.rnn.init_h(batch_size)
        log_probs = Variable(torch.zeros(batch_size, max_length))
        #log_probs = Variable(torch.zeros(batch_size))
        entropy = Variable(torch.zeros(batch_size))
        for step in range(seq_length):
            logits, h = self.rnn(x[:, step], h)
            log_prob = F.log_softmax(logits)
            prob = F.softmax(logits)
            #log_probs += NLLLoss(log_prob, target[:, step])
            log_probs[:, step] = NLLLoss(log_prob, target[:, step])
            entropy += -torch.sum((log_prob * prob), 1)
        return log_probs, entropy
        
    def sample(self, distributions = None, max_length=140):
        """
            Sample a batch of sequences

            Args:
                batch_size : Number of sequences to sample 
                max_length:  Maximum length of the sequences

            Outputs:
            seqs: (batch_size, seq_length) The sampled sequences.
            log_probs : (batch_size) Log likelihood for each sequence.
            entropy: (batch_size) The entropies for the sequences. Not
                                    currently used.
                                    
        """
        pattern = self.pattern
        pattern = tokenize_custom(pattern)

        start_token = Variable(torch.zeros(1).long())
        start_token[:] = self.voc.vocab['GO']
        
        h = self.rnn.init_h(1)
        x = start_token

        sequences = []
        log_probs = Variable(torch.zeros(1))
        finished = torch.zeros(1).byte()
        entropy = Variable(torch.zeros(1))
        if torch.cuda.is_available():
            finished = finished.cuda()
        
        tracker = 0
        opened = False
        opened_with_fragment = False
        linear_fragment = False
        free_fragment = False
        one_piece_fragment = False
        constrained_choice = False
        n_distribution = 0 
        log_probs_fragments = []
        for step in range(max_length):
           
            if pattern[tracker] == '@' and pattern[tracker-1]=='(':
                opened_with_fragment = True
                tracker += 1           
                
            if pattern[tracker] == '*' and pattern[tracker-1]=='(':
                opened = True
                tracker += 1
                
            if pattern[tracker] == '_':
                linear_fragment = True
                
            if pattern[tracker] == '@':
                one_piece_fragment = True
            
            if pattern[tracker].startswith('[') and ',' in pattern[tracker]:
                choices = pattern[tracker][1:-1].split(',')
                constrained_choice = True
                
            if pattern[tracker] == '*' and pattern[tracker-1]!='(':
                free_fragment = True
                
            if not opened and not linear_fragment and not free_fragment and not one_piece_fragment and not constrained_choice and not opened_with_fragment:

                logits, h = self.rnn(x, h)
                prob = F.softmax(logits)
                log_prob = F.log_softmax(logits)
                x = torch.multinomial(prob, num_samples=1).view(-1)
                #if pattern[tracker] == '*':

                start_token = Variable(torch.zeros(1).long())
                start_token[:] = self.voc.vocab[pattern[tracker]]
                #h = self.rnn.init_h(1)
                x = start_token
                #if pattern[tracker] == '(':
                #    print(prob)

                sequences.append(x.view(-1, 1))
                log_probs +=  NLLLoss(log_prob, x)
                entropy += -torch.sum((log_prob * prob), 1)
                x = Variable(x.data)
                # EOS_sampled = (x == self.voc.vocab['EOS']).data
                
                tracker += 1
                EOS_sampled = (x == self.voc.vocab['EOS']).byte()
                finished = torch.ge(finished + EOS_sampled, 1)
                if torch.prod(finished) == 1: break
              
            if constrained_choice:
                
                log_probs_fragment = Variable(torch.zeros(1))
                
                logits, h = self.rnn(x, h)
                prob = F.softmax(logits)
                log_prob = F.log_softmax(logits)
                mask = torch.zeros_like(prob)
                for choice in choices:      
                    mask[0, self.voc.vocab[choice]] = 1
                prob *= mask
                prob /= torch.sum(prob, dim=-1)
                x = torch.multinomial(prob, num_samples=1).view(-1)
                sequences.append(x.view(-1, 1))
                log_probs +=  NLLLoss(log_prob, x)
                log_probs_fragment +=  NLLLoss(log_prob, x)
                entropy += -torch.sum((log_prob * prob), 1)
                     
                tracker += 1
                constrained_choice = False
                log_probs_fragments.append(log_probs_fragment)
                
            if linear_fragment:
                if distributions:
                    distribution, loc, scale = distributions[n_distribution]
                    n_distribution += 1
                else:
                    distribution, loc, scale = 'uniform', 0, 5
                if distribution == 'normal':
                    flip = np.random.normal(loc, scale, size=1)
                    n_atoms = int(np.round(flip)[0])   
                elif distribution == 'uniform':
                    flip = np.random.uniform(loc, scale, size=1)
                    n_atoms = int(np.round(flip)[0])
                
                log_probs_fragment = Variable(torch.zeros(1))
                for i in range(n_atoms):
                    logits, h = self.rnn(x, h)
                    prob = F.softmax(logits)
                    log_prob = F.log_softmax(logits)
                    mask = torch.zeros_like(prob)
                    mask[0, self.voc.vocab['C']] = 1
                    mask[0, self.voc.vocab['N']] = 1
                    mask[0, self.voc.vocab['O']] = 1
                    prob *= mask
                    prob /= torch.sum(prob, dim=-1)
                    x = torch.multinomial(prob, num_samples=1).view(-1)
                    sequences.append(x.view(-1, 1))
                    log_probs +=  NLLLoss(log_prob, x)
                    log_probs_fragment +=  NLLLoss(log_prob, x)
                    entropy += -torch.sum((log_prob * prob), 1)
                     
                tracker += 1
                linear_fragment = False
                log_probs_fragments.append(log_probs_fragment)
                
            if free_fragment:
                ready_to_stop = True
                stop = False
                
                if distributions:
                    distribution, loc, scale = distributions[n_distribution]
                    n_distribution += 1
                else:
                    distribution, loc, scale = 'uniform', 0, 5
                if distribution == 'normal':
                    flip = np.random.normal(loc, scale, size=1)
                    n_atoms = int(np.round(flip)[0]) 
                elif distribution == 'uniform':
                    flip = np.random.uniform(loc, scale, size=1)
                    n_atoms = int(np.round(flip)[0])
                    
                count = 0   
                log_probs_fragment = Variable(torch.zeros(1))
                n_opened = 0 
                n_closed = 0
                cycle_opened = False
                cycle_closed = False
                while not stop:
                    logits, h = self.rnn(x, h)
                    prob = F.softmax(logits)
                    log_prob = F.log_softmax(logits)
                    mask = torch.zeros_like(prob)
                    x = torch.multinomial(prob, num_samples=1).view(-1)
                    sequences.append(x.view(-1, 1))
                    log_probs +=  NLLLoss(log_prob, x)
                    entropy += -torch.sum((log_prob * prob), 1)
                    
                    if (x == self.voc.vocab['(']).byte():
                        n_opened += 1
                    if (x == self.voc.vocab[')']).byte():
                        n_closed += 1
                    if (x == self.voc.vocab['1']).byte():
                        if cycle_opened:
                            cycle_closed = True
                        elif not cycle_opened:
                            cycle_opened = True
                    ready_to_stop = (n_opened==n_closed)
                    if ready_to_stop and count>= n_atoms  and (cycle_opened==cycle_closed):
                        stop = True
                    count += 1
                    
                tracker += 1
                free_fragment = False
                log_probs_fragments.append(log_probs_fragment)
                    
            if one_piece_fragment:
                ready_to_stop = True
                stop = False
                
                if distributions:
                    distribution, loc, scale = distributions[n_distribution]
                    n_distribution += 1
                else:
                    distribution, loc, scale = 'uniform', 0, 5
                if distribution == 'normal':
                    flip = np.random.normal(loc, scale, size=1)
                    n_atoms = int(np.round(flip)[0]) 
                elif distribution == 'uniform':
                    flip = np.random.uniform(loc, scale, size=1)
                    n_atoms = int(np.round(flip)[0])
                    
                count = 0   
                log_probs_fragment = Variable(torch.zeros(1))
                n_opened = 0 
                n_closed = 0
                cycle_opened = False
                cycle_closed = False
                while not stop:
                    previous_h = h
                    logits, h = self.rnn(x, h)
                 
                    prob = F.softmax(logits)
         
                    log_prob = F.log_softmax(logits)
                    x = torch.multinomial(prob, num_samples=1).view(-1)
                    if (x == self.voc.vocab['2']).byte():
                        h = previous_h
                        break
                    sequences.append(x.view(-1, 1))
                    log_probs +=  NLLLoss(log_prob, x)
                    entropy += -torch.sum((log_prob * prob), 1)
                    
                    if (x == self.voc.vocab['(']).byte():
                        n_opened += 1
                    if (x == self.voc.vocab[')']).byte():
                        n_closed += 1
                    if (x == self.voc.vocab['1']).byte():
                        if cycle_opened:
                            cycle_closed = True
                        elif not cycle_opened:
                            cycle_opened = True
                    ready_to_stop = (n_opened==n_closed)
                    if ready_to_stop and count>= n_atoms  and (cycle_opened==cycle_closed):
                        stop = True
                    count += 1
                    
                tracker += 1
                one_piece_fragment = False
                log_probs_fragments.append(log_probs_fragment)
                
            if opened_with_fragment:
                while pattern[tracker]!= '*':
                    logits, h = self.rnn(x, h)
                    prob = F.softmax(logits)
                    log_prob = F.log_softmax(logits)
                    start_token = Variable(torch.zeros(1).long())
                    start_token[:] = self.voc.vocab[pattern[tracker]]
                    x = start_token
                    sequences.append(x.view(-1, 1))
                    log_probs +=  NLLLoss(log_prob, x)
                    entropy += -torch.sum((log_prob * prob), 1)
                    x = Variable(x.data)
                    tracker += 1
              
                opened = True
                tracker += 1
                opened_with_fragment = False
                
            if opened:
                n_opened = 1
                n_closed = 0
                n_steps = 0
                log_probs_fragment = Variable(torch.zeros(1))
                while n_opened > n_closed:
                   # print(n_opened)
                   # print(n_closed)
                    
                    logits, h = self.rnn(x, h)
                    prob = F.softmax(logits)
                    log_prob = F.log_softmax(logits)
                    if False: #n_steps<5 and (n_opened-n_closed)==1:
                        #closing = True
                        #while closing: 
                        prob[0, self.voc.vocab[')']] = 0
                        
                        x = torch.multinomial(torch.softmax(prob, dim=-1), num_samples=1).view(-1)
                    else:        
                        x = torch.multinomial(prob, num_samples=1).view(-1)
                    #print(prob)
                    #print(x)
                    sequences.append(x.view(-1, 1))
                    log_probs +=  NLLLoss(log_prob, x)
                    log_probs_fragment += NLLLoss(log_prob, x)
                    entropy += -torch.sum((log_prob * prob), 1)
                    if (x == self.voc.vocab['(']).byte():
                        n_opened += 1
                    if (x == self.voc.vocab[')']).byte():
                        n_closed += 1
                    n_steps += 1
                    if n_steps>50:
                        break
                log_probs_fragments.append(log_probs_fragment)
                tracker += 1
                opened = False            
        sequences = torch.cat(sequences, 1)
        return sequences.data, log_probs, entropy, log_probs_fragments

def NLLLoss(inputs, targets):
    """
        Custom Negative Log Likelihood loss that returns loss per example,
        rather than for the entire batch.

        Args:
            inputs : (batch_size, num_classes) *Log probabilities of each class*
            targets: (batch_size) *Target class index*

        Outputs:
            loss : (batch_size) *Loss for each example*
    """

    if torch.cuda.is_available():
        target_expanded = torch.zeros(inputs.size()).cuda()
    else:
        target_expanded = torch.zeros(inputs.size())

    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
    loss = Variable(target_expanded) * inputs
    loss = torch.sum(loss, 1)
    return loss
