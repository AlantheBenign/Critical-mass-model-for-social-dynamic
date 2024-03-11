import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from IPython import display
import time
from scipy.signal import convolve2d
import os

# CONSTANTS
merchant_value = -1
vacancy_value = 0
resident_value = 1

# square neighborhood
# 1:= 3x3; 2:= 5x5; 3:= 7x7; ...
neighborhood = 2
kernel_side = (2*neighborhood+1)
max_neighbors = kernel_side**2 - 1

# densities
minimal_neighbors_resident_density = 0.25
maximal_neighbors_resident_density = 0.8
minimal_neighbors_merchant_density = 0.6
minimal_merchant_neighbors_resident_density = 0.2

# conditions of dissatisfaction Resident
minimal_neighbors_resident = np.round( max_neighbors * minimal_neighbors_resident_density )
maximal_neighbors_resident = np.round( max_neighbors * maximal_neighbors_resident_density )

minimal_merchant_neighbors_resident = np.round(max_neighbors * minimal_merchant_neighbors_resident_density)
maximal_merchant_neighbors_resident = max_neighbors

# conditions of dissatisfaction Merchant
minimal_neighbors_merchant = np.round( max_neighbors * minimal_neighbors_merchant_density )
maximal_neighbors_merchant = max_neighbors

minimal_resident_neighbors_merchant = 0
maximal_resident_neighbors_merchant = max_neighbors



# FUNCTIONS FOR PLOTTING

def plot_grid(agents, grid = False):
    # creates a discrete colormap
    vacancy = np.array([147.0/255, 148.0/255, 150.0/255])  # grey
    agent =  np.array([255.0/255, 255.0/255, 0.0/255])     # red 
    cyan =  np.array([0.0/255, 200.0/255, 255.0/255])      # cyan
    cmap = colors.ListedColormap([cyan, vacancy, agent])
    # determines the limits of each color:
    bounds = [merchant_value, vacancy_value, resident_value, resident_value + 1]            
    norm = colors.BoundaryNorm(bounds, cmap.N)

    size = 8
    fig, ax = plt.subplots(figsize=(size,size))
    ax.imshow(agents, cmap=cmap, norm=norm)

    # hide axis values
    plt.xticks([])  
    plt.yticks([])  

    # draws gridlines
    if grid:
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(-0.5, agents.shape[1]))
        ax.set_yticks(np.arange(-0.5, agents.shape[0]))

    plt.show()
    plt.close()


def save_image(agents, grid = False, fileName = "city"):
    # creates a discrete colormap
    vacancy = np.array([147.0/255, 148.0/255, 150.0/255])  # grey
    agent =  np.array([255.0/255, 255.0/255, 0.0/255])     # red 
    cyan =  np.array([0.0/255, 200.0/255, 255.0/255])      # cyan
    cmap = colors.ListedColormap([cyan, vacancy, agent])
    # determines the limits of each color:
    bounds = [merchant_value, vacancy_value, resident_value, resident_value + 1]            
    norm = colors.BoundaryNorm(bounds, cmap.N)

    size = 8
    fig, ax = plt.subplots(figsize=(size,size))
    ax.imshow(agents, cmap=cmap, norm=norm)

    # hide axis values
    plt.xticks([])  
    plt.yticks([])  

    # draws gridlines
    if grid:
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(-0.5, agents.shape[1]))
        ax.set_yticks(np.arange(-0.5, agents.shape[0]))

    plt.savefig(fileName + ".png")
    plt.close()


def plot_generic_grid(matrix, grid = False):
    n,m = np.array(matrix).shape
    colorDict = dict()
    for i in range(n):
        for j in range(m):
            if matrix[i][j] not in colorDict:
                rnd1 = np.random.random()
                rnd2 = np.random.random()
                rnd3 = np.random.random()
    
                color = np.array([rnd1,rnd2,rnd3])
                
                colorDict[matrix[i][j]] = color

    colorArray = np.zeros((len(colorDict),3))
    i = 0
    for color in list(colorDict.values()):
        colorArray[i] = color
        i += 1
        
    cmap = colors.ListedColormap(colorArray)
    # determines the limits of each color:
    bounds = np.array(sorted(list(colorDict.keys())), dtype = float)
    bounds = np.append(bounds, bounds[-1] + 1)

    # by hand correction for better visualization
    for i in range(len(bounds)):
        bounds[i] = bounds[i] - 0.5
        
    norm = colors.BoundaryNorm(bounds, cmap.N)

    size = 8
    fig, ax = plt.subplots(figsize=(size,size))
    ax.imshow(matrix, cmap=cmap, norm=norm)

    # hide axis values
    plt.xticks([])  
    plt.yticks([])  

    # draws gridlines
    if grid:
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(-0.5, agents.shape[1]))
        ax.set_yticks(np.arange(-0.5, agents.shape[0]))

    plt.show()
    plt.close()



#PROJECT CODE
def create_city(N, agentDensity, residentRelativeDensity, plot = True):
    # creates a city without borders, that will be a toroidal grid
    city = np.zeros(N*N, dtype=np.int16)
    agents = int(agentDensity*N*N)
    residents = int(residentRelativeDensity*agents)
    merchants = agents - residents

    # insert agents according to their densities
    for i in range(N*N):
        if i < residents:
            city[i] = resident_value
        elif residents <= i < residents + merchants:
            city[i] = merchant_value
        else:
            break

    # shuffle agents places
    np.random.shuffle(city)
    # reshape city array to matrix
    city = city.reshape((N,N)) 

    if plot:
        plot_grid(city)
    
    return city


# code by LUCA MINGARELLI https://lucamingarelli.com/Teaching/schelling.html
KERNEL = np.ones((kernel_side, kernel_side), dtype=np.int8)
middle = int(kernel_side/2)
KERNEL[middle][middle] = 0

def evolve(M, boundary='wrap'):
    kws = dict(mode='same', boundary=boundary)
    Resident_neighs = convolve2d(M == resident_value,  KERNEL, **kws)
    Merchant_neighs = convolve2d(M == merchant_value,  KERNEL, **kws)
    Neighs = convolve2d(M != vacancy_value,  KERNEL, **kws)

    # conditions of dissatisfaction
    Resident_dissatisfied = ((((Neighs < minimal_neighbors_resident) | (Neighs > maximal_neighbors_resident)) |                                  # minimal and maximal neighbors                   
                           ((Merchant_neighs < minimal_merchant_neighbors_resident) | (Merchant_neighs > maximal_merchant_neighbors_resident))) & # minimal and maximal merchant neighbors 
                           (M == resident_value))
    Merchant_dissatisfied = ((((Neighs < minimal_neighbors_merchant) | (Neighs > maximal_neighbors_merchant)) |                                  # minimal and maximal neighbors   
                           ((Resident_neighs < minimal_resident_neighbors_merchant) | (Resident_neighs > maximal_resident_neighbors_merchant))) & # minimal and maximal resident neighbors 
                           (M == merchant_value))
    M[Resident_dissatisfied | Merchant_dissatisfied] = vacancy_value
    vacant = (M == vacancy_value).sum()

    N_Resident_dissatisfied, N_Merchant_dissatisfied = Resident_dissatisfied.sum(), Merchant_dissatisfied.sum()
    filling = np.full(vacant, vacancy_value, dtype=np.int8)
    filling[:N_Resident_dissatisfied] = resident_value
    filling[N_Resident_dissatisfied:N_Resident_dissatisfied + N_Merchant_dissatisfied] = merchant_value
    np.random.shuffle(filling)
    M[M==vacancy_value] = filling



def simulate(city, timeSteps = 1000, plot = False, plotEnd = True, saveImage = False, fileName = "city"):
    for i in range(timeSteps):
        # stop condition
        if i%int(timeSteps/10) == 0:
            oldCity = np.copy(city)
            evolve(city)
            dif = oldCity - city
            if all(all(element == 0) for element in dif):
                break

        else:
            evolve(city)

        # time steps in which the city is plotted
        step = 100
        if i != 0 and i%step == 0 and plot:
            plot_grid(city)
            time.sleep(0.1)
            display.clear_output(wait=True)
                
    if plotEnd:
        display.clear_output(wait=True)
        plot_grid(city)

    if saveImage:
        save_image(city, fileName = fileName)


# CLUSTERING (MY DEFINITION)
def clustering_level(M, boundary='wrap'):
    kws = dict(mode='same', boundary=boundary)
    Resident_neighs = convolve2d(M == resident_value,  KERNEL, **kws)
    Merchant_neighs = convolve2d(M == merchant_value,  KERNEL, **kws)
    Neighs = convolve2d(M != vacancy_value,  KERNEL, **kws)
    
    # calculate clustering level of merchants
    side = M.shape[0]
    clustering = 0
    agents_with_neighs = 0

    for i in range(side):
        for j in range(side):
            if M[i][j] != vacancy_value and Neighs[i][j] != 0:
                agents_with_neighs += 1
                if M[i][j] == resident_value:
                    clustering += Resident_neighs[i][j]/(Neighs[i][j]*residentRelativeDensity)
                elif M[i][j] == merchant_value:
                    clustering += Merchant_neighs[i][j]/(Neighs[i][j]*(1-residentRelativeDensity))
                
    clustering = clustering/agents_with_neighs
    
    return clustering


# Hoshen-Kopelman Algorithm
class HoshenKopelman():

    def __init__(self, matrix):
        self.matrix = matrix
        self.labelsArray = np.arange(0,np.size(matrix) + 1, dtype = np.int16)
        self.labels = np.zeros(matrix.shape, dtype = np.int16)
        self.algorithmRan = False 
        
    def find(self, i):
        j = i
    
        while self.labelsArray[j] != j:
            j = self.labelsArray[j]
    
        return j
    
    def union(self, i, j):  
        index = self.find(j)
        value = self.find(i)
        
        self.labelsArray[index] = value
        
    def hk(self, target_value):
        n, m = self.matrix.shape
        largest_label = 0

        for j in range(m):
            for i in range(n):
                if self.matrix[i][j] == target_value:
                    if i > 0: up = self.matrix[i-1][j] 
                    else: up = None
    
                    if j > 0: left = self.matrix[i][j-1]
                    else: left = None
    
                    if left != target_value and up != target_value:
                        largest_label += 1
                        self.labels[i][j] = largest_label
                    
                    elif left == target_value and up != target_value:
                        self.labels[i][j] = self.find(self.labels[i][j-1])
    
                    elif left != target_value and up == target_value:
                        self.labels[i][j] = self.find(self.labels[i-1][j])
    
                    else:
                        smaller = min(self.labels[i][j-1], self.labels[i-1][j])
                        bigger = max(self.labels[i][j-1], self.labels[i-1][j])
                        self.union(smaller, bigger)
                        self.labels[i][j] = self.find(smaller)
    
        for j in range(m):
            for i in range(n):
                if self.labels[i][j]:
                    self.labels[i][j] = self.find(self.labels[i][j])

        self.algorithmRan = True
        
        return self.labels

    def cluster_data(self):
        if self.algorithmRan:
            n, m = self.labels.shape
            clustersLabels = set()
            
            for i in range(n):
                for j in range(m):
                    if self.labels[i][j] not in clustersLabels and self.labels[i][j] != 0:
                        clustersLabels.add(self.labels[i][j])
                        
            numClusters = len(clustersLabels)
            clustersSizes = dict()

            for label in clustersLabels:
                size = (self.labels == label).sum()
                clustersSizes[label] = size

            return numClusters, clustersSizes
            
        else:
            print('Run the algorithm first')    


