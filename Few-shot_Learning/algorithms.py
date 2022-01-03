####################################################################
# Implement the two techniques Finetuning and Reptile
####################################################################
import matplotlib.pyplot as plt
import torch
import numpy as np
from copy import deepcopy
from data_loader import SineLoader
from networks import SineNetwork

#Plotting Parameters
HORIZON = 10
TRAIN_C = (255/255, 165/255, 0)
TRAIN_T = (TRAIN_C[0],TRAIN_C[1],TRAIN_C[2],0.5)
TEST_C = (0, 1, 1)
TEST_T = (TEST_C[0],TEST_C[1],TEST_C[2],0.5)
ITER_S = ':'
MEAN_S = '-'

class Finetuning:
    def __init__(self, iterations, samples, batch_size, support_size, query_size, step):
        self.iterations = iterations
        self.samples = samples
        self.batch_size = batch_size
        self.support_size = support_size
        self.query_size = query_size
        self.step = step

    def plot_fn(self, ax, style, label, color=None):
        fx = np.arange(-5.0, 5.0, 0.1)
        fy = self.model(torch.Tensor(np.asarray(fx)).reshape(len(fx), 1)).detach().numpy()
        if color == None:
            ax.plot(fx, fy, style, label=label)
        else:
            ax.plot(fx, fy, style, label=label, color=color)

    def train_batch(self, x, y):
        # Updates the model on a single batch using SGD
        self.model.zero_grad()
        pred = self.model(x)
        loss = self.model.criterion(pred, y)
        loss.backward()
        for param in self.model.parameters():
            param.data -= self.step * param.grad.data

    def freeze_model(self, state):
        for layer in self.model.model['features'].children():
            for param in layer.parameters():
                param.requires_grad = state
        if state:
            self.model.model['out'].reset_parameters()

    def run(self, eval=100):
        self.model = SineNetwork()
        rng = np.random.RandomState(0)

        # Data Loaders
        train_loader = SineLoader(k=self.samples, k_test=0).generator(episodic=False, batch_size=self.samples, mode="train", reset_ptr=True)
        test_loader = SineLoader(k=self.support_size, k_test=self.query_size).generator(episodic=True, batch_size=None, mode="test", reset_ptr=True)

        _, (ax1, ax2) = plt.subplots(1, 2)

        iters = []
        train_losses = []
        train_means = []
        test_losses = []
        test_means = []
        for iteration in range(self.iterations):
            # Train on a batch of sine data
            x_batch, y_batch, _, _ = next(train_loader)
    
            # Train over the entire distribution of sine wave tasks for fine-tuning
            indices = rng.permutation(len(x_batch))
            for start in range(0, len(x_batch), self.batch_size):
                gather = indices[start:start+self.batch_size]
                self.train_batch(x_batch[gather], y_batch[gather])
                    
            # Evaluate on a new unseen task and plot results
            if iteration==0 or (iteration+1) % eval == 0:
                ax1.cla()
                ax2.cla()
                iters.append(iteration)

                # Keep weights for restoring later
                weights_before = deepcopy(self.model.state_dict())
                self.freeze_model(True)

                # Test the model for a fixed number of training epochs on the new task and evaluate against query set
                x_support, y_support, x_query, y_query = next(test_loader)
                train_losses.append(self.model.criterion(self.model(x_query), y_query).item())
                self.plot_fn(ax1, "--", "Initial Model", (0,0,0))
                for inneriter in range(32):
                    self.train_batch(x_support, y_support)
                    if (inneriter+1) % 8 == 0:
                        frac = (inneriter+1) / 32
                        self.plot_fn(ax1, "-", "Model at %i"%(inneriter+1), (1-frac, frac, 0, frac))
                test_losses.append(self.model.criterion(self.model(x_query), y_query).item())

                train_means.append(np.mean(train_losses[-HORIZON:]))
                test_means.append(np.mean(test_losses[-HORIZON:]))

                ax1.plot(x_support, y_support, "*", label="Support", color=(1,0,0))
                ax1.plot(x_query, y_query, "*", label="Query", color=(0,0,1))
                ax1.set_ylim(-5,5)
                ax1.legend(loc='upper right', fancybox=True, shadow=True)
                ax1.title.set_text("Fine-Tuning Transfer")
                ax1.set_xlabel("X")
                ax1.set_ylabel("Y")

                ax2.plot(iters[1:], train_losses[1:], ITER_S, label="Train Loss", color=TRAIN_T)
                ax2.plot(iters[1:], train_means[1:], MEAN_S, label="Train Mean", color=TRAIN_C)
                ax2.plot(iters[1:], test_losses[1:], ITER_S, label="Test Loss", color=TEST_T)
                ax2.plot(iters[1:], test_means[1:], MEAN_S, label="Test Mean", color=TEST_C)
                ax2.legend(loc="upper right", fancybox=True, shadow=True)
                ax2.title.set_text("Learning Curves")
                ax2.set_xlabel("Iterations")
                ax2.set_ylabel("MSE")
                
                plt.pause(0.01)
                print(f"-----------------------------")
                print(f"Iteration {iteration+1}")
                print("Before Transfer |", f"Train Loss: {train_losses[-1]:.3f}", f"Mean: {train_means[-1]:.3f}")
                print("After Transfer |", f"Test Loss: {test_losses[-1]:.3f}", f"Mean: {test_means[-1]:.3f}")

                # Restore weights from before testing
                self.freeze_model(False)
                self.model.load_state_dict(weights_before)
        
        
class Reptile:
    def __init__(self, iterations, samples, batch_size, support_size, query_size, outerstep, innerstep, innerepochs):
        self.iterations = iterations
        self.samples = samples
        self.batch_size = batch_size
        self.support_size = support_size
        self.query_size = query_size
        self.outerstep = outerstep
        self.innerstep = innerstep
        self.innerepochs = innerepochs

    def plot_fn(self, ax, style, label, color=None):
        fx = np.arange(-5.0, 5.0, 0.1)
        fy = self.model(torch.Tensor(np.asarray(fx)).reshape(len(fx), 1)).detach().numpy()
        if color == None:
            ax.plot(fx, fy, style, label=label)
        else:
            ax.plot(fx, fy, style, label=label, color=color)

    def train_batch(self, x, y):
        # Updates the model on a single batch using SGD
        self.model.zero_grad()
        pred = self.model(x)
        loss = self.model.criterion(pred, y)
        loss.backward()
        for param in self.model.parameters():
            param.data -= self.innerstep * param.grad.data

    def run(self, eval=100):
        self.model = SineNetwork()
        rng = np.random.RandomState(0)

        # Data Loaders
        train_loader = SineLoader(k=self.samples, k_test=0).generator(episodic=True, batch_size=None, mode="train", reset_ptr=True)
        test_loader = SineLoader(k=self.support_size, k_test=self.query_size).generator(episodic=True, batch_size=None, mode="test", reset_ptr=True)

        _, (ax1, ax2) = plt.subplots(1, 2)

        iters = []
        train_losses = []
        train_means = []
        test_losses = []
        test_means = []
        for iteration in range(self.iterations):
            # Train on a batch of sine data
            # Keep current weights for outerloop update
            weights_before = deepcopy(self.model.state_dict())               
            # Train over a single task of sine waves at each inner epoch
            for _ in range(self.innerepochs):                                
                x_batch, y_batch, _, _ = next(train_loader)                  
                indices = rng.permutation(len(x_batch)) 
                # get minibatchs for inner loop                     
                for start in range(0, len(x_batch), self.batch_size):        
                    gather = indices[start:start+self.batch_size]            
                    self.train_batch(x_batch[gather], y_batch[gather])       
            # Get new weights for outerloop update
            weights_after = self.model.state_dict()                          
            
            # performs an outerloop weight update separately from the innerloop training
            # re-scale the learning update
            outerstepsize = self.outerstep * (1 - iteration/self.iterations) 
            # interpolating old and new parameters
            self.model.load_state_dict({name :                               
                weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize  
                for name in weights_before
                })
                    
            # Evaluate on a new unseen task and plot results
            if iteration==0 or (iteration+1) % eval == 0:
                ax1.cla()
                ax2.cla()
                iters.append(iteration)

                # Keep weights for restoring later
                weights_before = deepcopy(self.model.state_dict())

                # Test the model for a fixed number of training epochs on the new task and evaluate against query set
                x_support, y_support, x_query, y_query = next(test_loader)
                train_losses.append(self.model.criterion(self.model(x_query), y_query).item())
                self.plot_fn(ax1, "--", "Initial Model", (0,0,0))
                for inneriter in range(32):
                    self.train_batch(x_support, y_support)
                    if (inneriter+1) % 8 == 0:
                        frac = (inneriter+1) / 32
                        self.plot_fn(ax1, "-", "Model at %i"%(inneriter+1), (1-frac, frac, 0, frac))
                test_losses.append(self.model.criterion(self.model(x_query), y_query).item())

                train_means.append(np.mean(train_losses[-HORIZON:]))
                test_means.append(np.mean(test_losses[-HORIZON:]))

                ax1.plot(x_support, y_support, "*", label="Support", color=(1,0,0))
                ax1.plot(x_query, y_query, "*", label="Query", color=(0,0,1))
                ax1.set_ylim(-5,5)
                ax1.legend(loc='upper right', fancybox=True, shadow=True)
                ax1.title.set_text("Reptile Transfer")
                ax1.set_xlabel("X")
                ax1.set_ylabel("Y")
                
                ax2.plot(iters[1:], train_losses[1:], ITER_S, label="Train Loss", color=TRAIN_T)
                ax2.plot(iters[1:], train_means[1:], MEAN_S, label="Train Mean", color=TRAIN_C)
                ax2.plot(iters[1:], test_losses[1:], ITER_S, label="Test Loss", color=TEST_T)
                ax2.plot(iters[1:], test_means[1:], MEAN_S, label="Test Mean", color=TEST_C)
                ax2.legend(loc="upper right", fancybox=True, shadow=True)
                ax2.title.set_text("Learning Curves")
                ax2.set_xlabel("Iterations")
                ax2.set_ylabel("MSE")

                plt.pause(0.01)
                print(f"-----------------------------")
                print(f"Iteration {iteration+1}")
                print("Before Transfer |", f"Train Loss: {train_losses[-1]:.3f}", f"Mean: {train_means[-1]:.3f}")
                print("After Transfer |", f"Test Loss: {test_losses[-1]:.3f}", f"Mean: {test_means[-1]:.3f}")

                # Restore weights from before testing
                self.model.load_state_dict(weights_before)
