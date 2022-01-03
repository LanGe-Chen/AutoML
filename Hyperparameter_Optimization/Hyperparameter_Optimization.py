import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time
import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm # for PDF and CDF
from scipy.interpolate import BSpline, make_interp_spline

'''Parameter search space:'''
Param_C = [0.1, 2.0] # Regularization parameter. 0 is no regularization, the higher the stronger
Param_kernel = [[0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0]] #['linear', 'poly', 'rbf', 'sigmoid'] # Basis functions, ignoring 'precomputed'
Param_degree = [1, 9] # only for Polynomial Kernel
Param_gamma = [0, 1] # Binary, Kernel coefficient 'scale' or 'auto', only used by ‘rbf’, ‘poly’ and ‘sigmoid’
Param_coef0 = [0, 1] # t is only significant in ‘poly’ and ‘sigmoid’.
Param_shrinking = [0, 1] # True or False, whether to use the shrinking heuristic
Param_probability = [0, 1] # True or False, internally uses 5-fold cross-validation, slow!
'''Remaining parameters left as default'''

class Optimizer:
    def PrintParameters(self, parameters):
        print("C =", parameters[0])
        print("kernel =", self.MapToKernel(parameters[1:5]))
        print("degree =", parameters[5])
        print("gamma =", self.MapToScale(parameters[6]))
        print("coef0 =", parameters[7])
        print("shrink =", self.MapToBool(parameters[8]))
        print("probability =", self.MapToBool(parameters[9]))

    def MapToKernel(self, onehot):
        onehot = onehot.tolist()
        if onehot == Param_kernel[0]:
            return 'linear'
        if onehot == Param_kernel[1]:
            return 'poly'
        if onehot == Param_kernel[2]:
            return 'rbf'
        if onehot == Param_kernel[3]:
            return 'sigmoid'

    def MapToScale(self, param):
        return 'scale' if param==0 else 'auto'
    
    def MapToBool(self, param):
        return bool(param)

    # Build model from chosen parameters and calculate fitting score
    # When using 'Train' it may overfit, when using 'Test' the score may be low
    def Evaluate(self, parameters):
        # print("Evaluating Parameters:", parameters)
        model = SVC(
            C=parameters[0], 
            kernel=self.MapToKernel(parameters[1:5]), 
            degree=3 if parameters[5]==None else parameters[5], 
            gamma=self.MapToScale(parameters[6]), 
            coef0=parameters[7], 
            shrinking=self.MapToBool(parameters[8]), 
            probability=self.MapToBool(parameters[9]),
            random_state=SEED
        )
        model.fit(self.X_train, self.Y_train)
        y_pred = model.predict(self.x_test)
        return accuracy_score(self.y_test, y_pred)

    def SampleRandom(self):
        C = random.uniform(Param_C[0], Param_C[1])
        kernel = random.choice(Param_kernel)
        degree = random.randint(Param_degree[0], Param_degree[1])
        gamma = random.choice(Param_gamma)
        coef0 = random.uniform(Param_coef0[0], Param_coef0[1])
        shrinking = random.choice(Param_shrinking)
        probability = random.choice(Param_probability)
        params = [C]
        params.extend(kernel)
        params.extend([degree, gamma, coef0, shrinking, probability])
        return np.asarray(params)

    def SampleGrid(self, a, b, c, d, e, f, g):
        params = [a]
        params.extend(b)
        params.extend([c,d,e,f,g])
        return np.asarray(params)

    def OptimizeParameters(self, X, Y):
        if EVALUATION == 'Train':
            self.X_train = X
            self.Y_train = Y
            self.x_test = X
            self.y_test = Y
        if EVALUATION == 'Test':
            self.X_train, self.x_test, self.Y_train, self.y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=SEED)

        args = [0]
        values = [0]
        parameters = None
        score = 0
        count = 0

        if SEARCH_TYPE == 'Random':
            print("Optimizing hyperparameters via random search...")

            start = time.time()
            while time.time()-start < TIMEOUT:
                # Select Parameters
                _parameters_ = self.SampleRandom()

                # Evaluate model and obtain score
                _score_ = self.Evaluate(_parameters_)

                # Update parameters and best score
                if _score_ > score:
                    parameters = _parameters_
                    score = _score_
                
                count += 1
                args.append(time.time() - start)
                # args.append(count)
                values.append(score)
            
        if SEARCH_TYPE == 'Grid':
            C = np.arange(Param_C[0], Param_C[1]+(Param_C[1]-Param_C[0])/C_RESOLUTION, (Param_C[1]-Param_C[0])/C_RESOLUTION)
            kernel = Param_kernel
            degree = range(Param_degree[0], Param_degree[1]+DEGREE_STEP, DEGREE_STEP)
            gamma = Param_gamma
            coef0 = np.arange(Param_coef0[0], Param_coef0[1]+(Param_coef0[1]-Param_coef0[0])/COEF0_RESOLUTION, (Param_coef0[1]-Param_coef0[0])/COEF0_RESOLUTION)
            shrinking = Param_shrinking
            probability = Param_probability

            combinations = len(C) * len(kernel) * len(degree) * len(gamma) * len(coef0) * len(shrinking) * len(probability)
            print("Optimizing hyperparameters using", combinations, "combinations via grid search...")

            start = time.time()
            for a in C:
                for b in kernel:
                    for c in degree:
                        for d in gamma:
                            for e in coef0:
                                for f in shrinking:
                                    for g in probability:
                                        if TIMEOUT > 0 and time.time()-start > TIMEOUT:
                                            break
                                            
                                        _parameters_ = self.SampleGrid(a,b,c,d,e,f,g)
                                        _score_ = self.Evaluate(_parameters_)

                                        # Update parameters and best score
                                        if _score_ > score:
                                            parameters = _parameters_
                                            score = _score_

                                        count += 1
                                        args.append(time.time() - start)
                                        values.append(score)
                                        print("Progress:", round(100 * float(count) / float(combinations), 3), "%", end="\r")

        if SEARCH_TYPE == 'Bayesian':
            print("Optimizing hyperparameters via naive Bayesian search...")

            model = GaussianProcessRegressor()
            X = np.asarray([self.SampleRandom() for i in range(BAYESIAN_SAMPLES)])
            Y = np.asarray([self.Evaluate(i) for i in X])
            model.fit(X,Y)
            start = time.time()
            while TIMEOUT == 0 or time.time()-start < TIMEOUT:
                x = self.opt_acquisition(X, model, BAYESIAN_SAMPLES)
                y = self.Evaluate(x)
                estimate, _ = self.surrogate(model, [x])

                x = x.reshape(1,len(x))
                X = np.vstack([X, x])

                y = np.array(y).reshape(1,)
                Y = np.concatenate([Y, y])

                model.fit(X,Y)

                idx = np.argmax(Y)
                parameters, score = X[idx], Y[idx]

                args.append(time.time() - start)
                values.append(score)

        return parameters, score, args, values

    def surrogate(self, model, X):
        return model.predict(X, return_std=True)

    def acquisition(self, X, args, model):
        # Calculate the best surrogate score found so far
        yhat, _ = self.surrogate(model, X)
        best = max(yhat)
        # Calculate mean and stdev via surrogate function
        mu, std = self.surrogate(model, args)
        # Calculate the Probability of Improvement (PI)
        z = (best-mu)/std
        probs = (best-mu)*norm.cdf(z)+std*norm.pdf(z) # Expected Improvement (EI)
        return probs

    def opt_acquisition(self, X, model, samples):
        # Random search, generate random samples
        args = np.asarray([self.SampleRandom() for i in range(samples)])
        # Calculate the acquisition function for each sample
        scores = self.acquisition(X, args, model)
        # Locate the index of the largest scores
        ix = np.argmax(scores)
        return args[ix]

def Plot(x, y, label):
    muX = []
    muY = []
    stdY = []

    length = max([len(i) for i in x])
    for i in range(length):
        sum = 0
        for j in x:
            idx = min(i, len(j)-1)
            sum += j[idx]
        muX.append(sum/len(x))
    for i in range(length):
        sum = 0
        for j in y:
            idx = min(i, len(j)-1)
            sum += j[idx]
        muY.append(sum/len(y))
    for i in range(length):
        sum = 0
        for j in y:
            idx = min(i, len(j)-1)
            d = j[idx] - muY[idx]
            sum += d*d
        stdY.append(np.sqrt(sum/len(y)))

    muX = np.asarray(muX)

    muY = np.asarray(muY)

    stdY = np.asarray(stdY)

    plt.plot(muX, muY, '-', label=label)
    plt.fill_between(muX, muY - stdY, muY + stdY, alpha=0.1)

######PARAMETER SETTINGS######
SEED = random.randint(0, 10000000)
TIMEOUT = 10 # Timeout when to stop the search, where 0 is infinity.
EVALUATION = 'Test' # Type of accuracy evaluation ['Train', 'Test']
TEST_SIZE = 0.1 # Test set split in case of 'Test' evaluation type
SEARCH_TYPE = 'Random' # Type of search ['Grid', 'Random', 'Bayesian']
DATA_INDEX = 0 # Index of dataset [0,1,2,3,4]

C_RESOLUTION = 5 # Number of samples between min to max (uniform)
DEGREE_STEP = 2 # Step size between min to max
COEF0_RESOLUTION = 5 # Number of samples between min to max (uniform)
BAYESIAN_SAMPLES = 5 # Number of samples generated per iteration for naive Bayesian optimization

EXPERIMENT_TRIALS = 1 # Number of experiment runs to gather mean/std data for plotting
##############################

file = f"dataset-{DATA_INDEX}.npy" 
data = np.load(file) 
optimizer = Optimizer()

# Single Run
# params, score, _, _ = optimizer.OptimizeParameters(data[:,:-1], data[:,-1])
# print("Dataset:")
# print(file)
# print("")
# print("Search Type:")
# print(SEARCH_TYPE)
# print("")
# print("Best found score:")
# print(score)
# print("")
# print("Best found parameters:")
# optimizer.PrintParameters(params)
# print("")
# print("Train-Test Split:")
# print((1.0-TEST_SIZE)*100 if EVALUATION=='Test' else 100, "% train data")
# print(TEST_SIZE*100 if EVALUATION=='Test' else 0, "% test data")
# print("")

# Plotting
args = []
values = []

SEARCH_TYPE = 'Random'
for i in range(EXPERIMENT_TRIALS):
    print("Running trial",(i+1),"/",EXPERIMENT_TRIALS)
    _, _, a, v = optimizer.OptimizeParameters(data[:,:-1], data[:,-1])
    args.append(a)
    values.append(v)
Plot(args, values, SEARCH_TYPE)

SEARCH_TYPE = 'Grid'
for i in range(EXPERIMENT_TRIALS):
    print("Running trial",(i+1),"/",EXPERIMENT_TRIALS)
    _, _, a, v = optimizer.OptimizeParameters(data[:,:-1], data[:,-1])
    args.append(a)
    values.append(v)
Plot(args, values, SEARCH_TYPE)

SEARCH_TYPE = 'Bayesian'
for i in range(EXPERIMENT_TRIALS):
    print("Running trial",(i+1),"/",EXPERIMENT_TRIALS)
    _, _, a, v = optimizer.OptimizeParameters(data[:,:-1], data[:,-1])
    args.append(a)
    values.append(v)
Plot(args, values, SEARCH_TYPE)

plt.title("Hyperparameter Optimization - Dataset "+str(DATA_INDEX))
plt.xlabel("Time in s")
plt.ylabel("Accuracy")
plt.ylim([0,1])
plt.legend(loc = "upper left")
plt.show()