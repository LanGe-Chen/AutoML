# AutoML - Few-shot Learning

Deep neural networks have achieved enabled great successes in various areas. However, they are
notoriously data-hungry: they require a lot of data to achieve a good level of performance. Naturally,
this raises the question of how we can let them learn more quickly (from fewer data) just like
humans.

In this project, there are two approaches will be implemented and compared: i) transfer learning, and
ii) meta-learning. More specifically, pre-training & fine-tuning and Reptile for few-shot sine wave regression problem. 

## Prerequisites

Have Python3 and pytorch installed.

## Executing program

The project contains the following files:
* main.py is the main script to interact with the data loader and run for experiments.
* networks.py Code that contains the SineNetwork
* data loader.py The file containing the data loader
* algorithms.py contains fine-tuning and Reptile algorithms

```
Finetuning().run(eval=100)
# Reptile().run(eval=100) # comment the Finetuning() and comment out Reptile() to switch the algorithm
```

## Acknowledgments

The experiment results and discussion are shown in the 'Report_Few-shot_Learning.pdf'