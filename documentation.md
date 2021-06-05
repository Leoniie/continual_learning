
# Notes on run_replay_vib.py
Implementation of the vib model for more than one task (number of tasks to be learning can be changed with num_tasks in l. 115)). Model will be run only for one beta_1 and one beta_2, can be changed with the arguments --beta_1 and --beta_2. Need to make sure the dimensions of the encoder, classification decoder and reconstruction decoder match (same number of gaussian units, default is 100). 

*Would it make sense to initialize a new reconstruction decoder for every task?* Not really, since we don't label the training tasks in the replay buffer, therefore during training the model would not know which reconstruction decoder to use for the replay data.

Right now, the model trains on two tasks - the first task is learned without replay, then the replay buffer is filled and then the second task is learned. 

Implemented the option to freeze the encoder during training of the second task.

Implemented a "crash scenario": If loss explodes, prints "crash" but no system exit, so that experiment can continue with other parameter configuration (no perfect error handling...).

**Replay methods:** Simple replay, generative replay (and no replay by setting the replay weight to 0, and beta_1 and beta_2, too).

Parameters to variate: **beta_1, beta_2, learning rate,** replay_weight, n_replay



# Notes on run_vib.py

Implementation of the variational information bottleneck (VIB). No replay method, trains the model for one SINGLE task, but trying out various different beta_1 and beta_2. For each configuration, stores the best accuracy achieved and the epoch in which it was achieved. Helps to test different beta configurations to find out which work well and which don't. However, program might crash during execution because learning rate may need to be adapted to different sizes of beta. Therefore it's more efficient to try a smaller number of betas with an appropriate learning rate. 

# Notes on run_replay.py

**Arguments**
- I still have the arguments for the **"weight decay"** in the code but I am not using them. Should I keep it? What do we need them for?
- What about the random seed that is initialized in the beginning?

**Classifier**
- I  am keeping the option to use and store multiple classifiers, one for each task. It might be useful for debugging in experiments with different datasets.
- So far, all classifiers have been trained with the "sgd" optimizer. The code also has the options "adam" and "adagrad" --> Might make sense to try that out?

**Train**
- The combined loss of replay and training data is right now weighted as default 1:1. However, the weigthing can be changed easily with the input argument --replay_weight
- so far, simple replay uses soft cross entropy for calculating the loss on the replay data. I thought it makes sense if the model remembers not only how samples are classified but also how stronlg it recognizes other data. Maybe this is something we could change back to hard classification
- Although the input argument "criterion" is passed to the training functions, I currently use the same criterion for all tasks: torch.nn.CrossEntropyLoss() for original training data and softcrossentropy for replay data (both simple and generative)


### Generative replay


**VAE**
I am using exactly the same VAE as the pytorch example vae implementation.
It is symmetric and has two hidden layers both for the encoder and the decoder. The number of units in the hidden layers is at default 400, and the number of gaussian units is 100. Both can be changed in the input.
The default learning rate is 1e-3 and the default number of training epochs is 10.
The images produced by the VAE can only be human-checked for split mnist (perm mnist can't be solved by humans). Right now, the images get blurry.
I have tried implementing a third hidden layer for the encoder and decoder each, and it actually seemed as if the model was creating better images. However, we want to keep the VAE as small as possible, so I deleted the extra layers. By finding good learning parameters, it should be possibel to achieve good reults even with two layers.

- I implemented the generative model with one VAE per task first - this naturally achieves better results than one single VAE with the same learning parameters
- I have implemented a single VAE version (option can be chosen at input level): Performanc is worse, but not much!
- The loss function of the replay is currently binary cross entropy between the inputs and the outputs






## Ideas for moving on

 **Maximally Inferred Retrieval:**
- Training only on tasks that are ambigious (difficult training examples)
- Think of other methods to choose relevant training data

 **Deep Variational Bottleneck:**


 **Tiny episodic memory**
 - Take ideas from that paper, i.e.:
   - for simple replay, don't store single examples but samples that represent an average over the input, i.e. k-means
   - 

#leoniem
