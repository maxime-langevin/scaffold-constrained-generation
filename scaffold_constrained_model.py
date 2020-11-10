# Copyright Notice: Permission is hereby granted, free of charge, and for non-commercial use only, to any person # from academic research or non-profit organization obtaining a copy of this software and associated             # documentation files (the "Software"), to use, copy, modify, or merge the Software, subject to the following      # conditions: this permission notice shall be included in all copies or substantial portions of the Software. All  # other rights are reserved. The Software is provided 'as is", without warranty of any kind, express or implied,  # including the warranties of noninfringement.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_structs import tokenize
from utils import Variable, replace_halogen, tokenize_custom
from model import MultiGRU
import re
import random
import numpy as np
from scipy.stats import uniform

class scaffold_constrained_RNN():
    """Implements the scaffold constrained RNN sampler"""
    def __init__(self, voc):
        self.rnn = MultiGRU(voc.vocab_size)
        if torch.cuda.is_available():
            self.rnn.cuda()
        self.voc = voc

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
        entropy = Variable(torch.zeros(batch_size))
        for step in range(seq_length):
            logits, h = self.rnn(x[:, step], h)
            log_prob = F.log_softmax(logits)
            prob = F.softmax(logits)
            log_probs[:, step] = NLLLoss(log_prob, target[:, step])
            entropy += -torch.sum((log_prob * prob), 1)
        return log_probs, entropy
        

    
    def sample(self, pattern = "CC(*)CC", batch_size = 128, max_length=140):
        """ 
            Only difference with classic RNN based sampling.
            Sample a batch of sequences with given scaffold.

            Args:
                pattern: Scaffold that need to be respected
                distributions: Distribution on the length of 
                batch_size : Number of sequences to sample 
                max_length:  Maximum length of the sequences

            Outputs:
            seqs: (batch_size, seq_length) The sampled sequences.
            log_probs : (batch_size) Log likelihood for each sequence.
            entropy: (batch_size) The entropies for the sequences. Not
                                    currently used.
                                    
        """
        
        # get the tokenized version of the pattern
        pattern = np.array(tokenize_custom(pattern))

        start_token = Variable(torch.zeros(batch_size).long())
        start_token[:] = self.voc.vocab['GO']
        
        h = self.rnn.init_h(batch_size)
        x = start_token
        
        sequences = []
        log_probs = Variable(torch.zeros(batch_size))
        finished = torch.zeros(batch_size).byte()
        entropy = Variable(torch.zeros(batch_size))

        if torch.cuda.is_available():
            finished = finished.cuda()
        
        # tracks if there is an opened parenthesis
        opened = np.array(np.zeros(shape=batch_size), dtype=bool)
        
        # tracks if there is a constrained choice 
        constrained_choices = np.array(np.zeros(shape=batch_size), dtype=bool)
        
        # tracks number of opening and closing parentheses 
        opening_parentheses = np.zeros(shape=batch_size)
        closing_parentheses = np.zeros(shape=batch_size)
        
        # tracks number of steps in the fragment that is being sampled 
        # (if the RNN never samples the matching parenthesis we terminate sampling of this given molecule
        n_steps = np.zeros(shape=batch_size)
        
        # tracks opened cycles
        opened_cycles = [['A', ] for i in range(batch_size)]
        opened_cycles_outside_sampling = [[] for i in range(batch_size)]
        counts = np.zeros(shape=batch_size, dtype=int)
        
        # tracks the position in the scaffold's pattern
        trackers = np.zeros(shape=batch_size, dtype=int)
        current_pattern_indexes = np.array([pattern[index] for index in trackers])
        
        for step in range(max_length):
            
            # Getting the position in the pattern of every example in the batch
            previous_pattern_indexes = current_pattern_indexes
            current_pattern_indexes = np.array([pattern[index] for index in trackers])
            
            # Check if a decoration is currently opened
            opened = np.logical_or(np.logical_and(current_pattern_indexes=='*', previous_pattern_indexes=='('), opened)
            
            # And if we're heading to a constrained choice
            constrained_choices = np.array([x[0] == '['  and ',' in x for x in current_pattern_indexes], dtype=bool)
            
            # In this case we already sampled this branch and need to move on for one step in the pattern
            #trackers += 1 * np.logical_and(current_pattern_indexes=='*', previous_pattern_indexes=='(')
            
            # Sample according to conditional probability distribution of the RNN
            logits, h = self.rnn(x, h)
            prob = F.softmax(logits)
            log_prob = F.log_softmax(logits)
            x = torch.multinomial(prob, num_samples=1).view(-1)
            
            # If not opened, replace with current pattern token, else keep the sample
            # And update number of opened and closed parentheses
            # If closed, resume to opened
            
            # iterating over the batch: 
            # there might be a smart way to parallelize all this but we didn't focus on it
            # as sampling speed is not necessarly a bottleneck in our applications
            for i in range(batch_size):
                
                # to keep track of opening and closing parentheses 
                is_open = opened[i]
                        
                # if we have a constrained choice 
                # we apply a mask on the probability vector
                if constrained_choices[i]:
                    
                    choices = current_pattern_indexes[i][1:-1].split(',')
                    probabilities = prob[i, :]
                    mask = torch.zeros_like(probabilities)
                    for choice in choices:      
                        mask[self.voc.vocab[choice]] = 1
                    probabilities *= mask
                    probabilities /= torch.sum(probabilities, dim=-1)
                    x[i] = torch.multinomial(probabilities, num_samples=1).view(-1)
                    trackers[i] += 1 * (x[i] != self.voc.vocab['EOS']).byte()
                
                # In this case we need to sample
                # We make the distinction between branch (first case) and linked (second case)    
                elif current_pattern_indexes[i]=='*':
                    if pattern[trackers[i]+1] == ')':
                        n_steps[i] += 1
                        if n_steps[i]>50:
                            x[i] = self.voc.vocab['EOS']
                        opening_parentheses[i] += (x[i] == self.voc.vocab['(']).byte() * 1
                        closing_parentheses[i] += (x[i] == self.voc.vocab[')']).byte() * 1
                        n_opened = opening_parentheses[i]
                        n_closed = closing_parentheses[i]
                        
                        if (n_opened+1==n_closed):
                            opening_parentheses[i] += 1
                            opened[i] = False
                            trackers[i] += 2  
                    else:
                        # The following lines are to avoid that sampling finishes too early
                        probabilities = prob[i, :]
                        mask = torch.ones_like(probabilities)
                        mask[self.voc.vocab['EOS']] = 0
                        
                        # If cycle still opened outside the sampling, we need to avoid it
                        for cycle in opened_cycles_outside_sampling[i]:
                            mask[self.voc.vocab[str(cycle)]] = 0
                        probabilities *= mask
                        probabilities /= torch.sum(probabilities, dim=-1)
                        x[i] = torch.multinomial(probabilities, num_samples=1).view(-1)
                        
                        opening_parentheses[i] += (x[i] == self.voc.vocab['(']).byte() * 1
                        closing_parentheses[i] += (x[i] == self.voc.vocab[')']).byte() * 1
                        n_opened = opening_parentheses[i]
                        n_closed = closing_parentheses[i]
                        for cycle in range(1, 10):
                            if (x[i] == self.voc.vocab[str(cycle)]).byte() and (cycle in opened_cycles[i]):
                                opened_cycles[i].remove(cycle)
                                break
                            elif (x[i] == self.voc.vocab[str(cycle)]).byte():
                                opened_cycles[i].append(cycle)
                                break
                                
                        # Override with specified distribution for minimal fragment size
                        # You could also make this an argument of the sample function
                        # You can also keep this parameter fixed manually as it currently is
                        # The sampling of the linker will only stop when size is > to minimal_linked_size 
                        # and cycles and branches are completed
                        
                        minimal_linker_size = 5
                        
                        if (n_opened==n_closed) and len(opened_cycles[i])==1 and counts[i]>minimal_linker_size:
                            opening_parentheses[i] += 1
                            opened[i] = False
                            trackers[i] += 1 
                        else:
                            counts[i] += 1                        
                
                # If we avoided all previous cases, then we do not sample and instead read the pattern
                else:
                    x[i] = self.voc.vocab[current_pattern_indexes[i]]
                    trackers[i] += 1 * (x[i] != self.voc.vocab['EOS']).byte()
                    #opening_parentheses[i] += (x[i] == self.voc.vocab['(']).byte() * 1
                    #closing_parentheses[i] += (x[i] == self.voc.vocab[')']).byte() * 1
                    if (x[i] == self.voc.vocab[')']).byte():
                        opened[i] = False
                        
                    for cycle in range(1, 10):
                        if (x[i] == self.voc.vocab[str(cycle)]).byte() and (cycle in opened_cycles_outside_sampling[i]):
                            opened_cycles_outside_sampling[i].remove(cycle)
                            break
                        elif (x[i] == self.voc.vocab[str(cycle)]).byte():
                            opened_cycles_outside_sampling[i].append(cycle)
                            break
                        
                 
            sequences.append(x.view(-1, 1))
            log_probs += NLLLoss(log_prob, x)
            entropy += -torch.sum((log_prob * prob), 1)

            x = Variable(x.data)
            EOS_sampled = (x == self.voc.vocab['EOS']).byte()
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1: break

        sequences = torch.cat(sequences, 1)
        return sequences.data, log_probs.data, entropy.data


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
