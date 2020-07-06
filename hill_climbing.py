import torch
import pickle
import numpy as np
import time
import os
from shutil import copyfile
import gc
from rdkit import Chem
from model import RNN
from scaffold_constrained_model import scaffold_constrained_RNN 
from data_structs import Vocabulary, Experience
from scoring_functions import get_scoring_function
from utils import Variable, seq_to_smiles, fraction_valid_smiles, unique
from vizard_logger import VizardLog

def hill_climbing(pattern=None, restore_agent_from='data/Prior.ckpt',
                scoring_function='tanimoto',
                scoring_function_kwargs=None,
                save_dir=None, learning_rate=0.0005,
                batch_size=64, n_steps=10,
                num_processes=0, use_custom_voc="data/Voc"):

    voc = Vocabulary(init_from_file=use_custom_voc)

    start_time = time.time()
    if pattern:
        Agent = scaffold_constrained_RNN(voc)
    else:
        Agent = RNN(voc)

    logger = VizardLog('data/logs')


    if torch.cuda.is_available():
        Agent.rnn.load_state_dict(torch.load(restore_agent_from))
    else:
        Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))



    optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=learning_rate)

    # Scoring_function
    scoring_function = get_scoring_function(scoring_function=scoring_function, num_processes=num_processes,
                                            **scoring_function_kwargs)

    # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
    # occur more often (which means the agent can get biased towards them). Using experience replay is
    # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
    experience = Experience(voc)

    # Log some network weights that can be dynamically plotted with the Vizard bokeh app
    logger.log(Agent.rnn.gru_2.weight_ih.cpu().data.numpy()[::100], "init_weight_GRU_layer_2_w_ih")
    logger.log(Agent.rnn.gru_2.weight_hh.cpu().data.numpy()[::100], "init_weight_GRU_layer_2_w_hh")
    logger.log(Agent.rnn.embedding.weight.cpu().data.numpy()[::30], "init_weight_GRU_embedding")
    logger.log(Agent.rnn.gru_2.bias_ih.cpu().data.numpy(), "init_weight_GRU_layer_2_b_ih")
    logger.log(Agent.rnn.gru_2.bias_hh.cpu().data.numpy(), "init_weight_GRU_layer_2_b_hh")

    # Information for the logger
    step_score = [[], []]

    print("Model initialized, starting training...")

    for step in range(n_steps):

        # Sample from Agent
        if pattern:
            seqs, agent_likelihood, entropy = Agent.sample(pattern, batch_size)
        else:
            seqs, agent_likelihood, entropy = Agent.sample(batch_size)
        gc.collect()
        # Remove duplicates, ie only consider unique seqs
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        entropy = entropy[unique_idxs]

        # Get prior likelihood and score
        smiles = seq_to_smiles(seqs, voc)
        score = scoring_function(smiles)
        
        new_experience = zip(smiles, score, agent_likelihood)
        experience.add_experience(new_experience)
        
        indexes = np.flip(np.argsort(np.array(score)))
        # Train the agent for 10 epochs on hill-climbing procedure
        for epoch in range(10):
            loss = Variable(torch.zeros(1))
            counter = 0
            seen_seqs = []
            for j in indexes:
                if counter>50:
                    break
                seq = seqs[j]
                s = smiles[j]
                if s not in seen_seqs:
                    seen_seqs.append(s)
                    log_p, _ = Agent.likelihood(Variable(seq).view(1, -1))
                    loss -= log_p.mean()
                    counter += 1
            loss /= counter
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print some information for this step
        time_elapsed = (time.time() - start_time) / 3600
        time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
        print("\n       Step {}   Fraction valid SMILES: {:4.1f}  Time elapsed: {:.2f}h Time left: {:.2f}h".format(
              step, fraction_valid_smiles(smiles) * 100, time_elapsed, time_left))
        print("  Agent    Prior   Target   Score             SMILES")
        for i in range(10):
            print(" {:6.2f}     {}".format(score[i],smiles[i]))
        # Need this for Vizard plotting
        step_score[0].append(step + 1)
        step_score[1].append(np.mean(score))

        # Log some weights
        logger.log(Agent.rnn.gru_2.weight_ih.cpu().data.numpy()[::100], "weight_GRU_layer_2_w_ih")
        logger.log(Agent.rnn.gru_2.weight_hh.cpu().data.numpy()[::100], "weight_GRU_layer_2_w_hh")
        logger.log(Agent.rnn.embedding.weight.cpu().data.numpy()[::30], "weight_GRU_embedding")
        logger.log(Agent.rnn.gru_2.bias_ih.cpu().data.numpy(), "weight_GRU_layer_2_b_ih")
        logger.log(Agent.rnn.gru_2.bias_hh.cpu().data.numpy(), "weight_GRU_layer_2_b_hh")
        logger.log("\n".join([smiles + "\t" + str(round(score, 2)) for smiles, score in zip \
                            (smiles[:12], score[:12])]), "SMILES", dtype="text", overwrite=True)
        logger.log(np.array(step_score), "Scores")

    # If the entire training finishes, we create a new folder where we save this python file
    # as well as some sampled sequences and the contents of the experinence (which are the highest
    # scored sequences seen during training)
    if not save_dir:
        save_dir = 'data/results/run_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    try:
        os.makedirs(save_dir)
    except:
        print("Folder already existing... overwriting previous results")
        
    copyfile('train_agent.py', os.path.join(save_dir, "train_agent.py"))

    experience.print_memory(os.path.join(save_dir, "memory"))
    torch.save(Agent.rnn.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))
    previous_smiles = []
    with open(os.path.join(save_dir, "memory.smi"), 'w') as f:
        for i, exp in enumerate(experience.memory):
             try:
                 if Chem.MolToSmiles(Chem.rdmolops.RemoveStereochemistry(Chem.MolFromSmiles(exp[0]))) not in previous_smiles:
                     f.write("{}\n".format(exp[0]))
                     previous_smiles.append(Chem.MolToSmiles(Chem.rdmolops.RemoveStereochemistry(Chem.MolFromSmiles(exp[0]))))
             except:
                 pass

if __name__ == "__main__":
    hill_climbing()
