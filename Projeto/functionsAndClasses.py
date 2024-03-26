import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from IPython import display
import time
from scipy.signal import convolve2d

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
    colorArray[0] = [147.0/255, 148.0/255, 150.0/255]
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
    
class City():

    merchant_value = -1
    vacancy_value = 0
    resident_value = 1
    
    # square neighborhood
    # 1:= 3x3; 2:= 5x5; 3:= 7x7; ...
    neighborhood = 1
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

    def __init__(self, *args):
        if len(args) == 3:
            N, agentDensity, residentRelativeDensity = args
            self.city = np.zeros(N*N, dtype=np.int8)
            agents = int(agentDensity*N*N)
            residents = int(residentRelativeDensity*agents)
            merchants = agents - residents

            self.residentsArray = np.zeros(residents, dtype = 'i, i')
            self.merchantsArray = np.zeros(merchants, dtype = 'i, i')
            self.vacantArray = np.zeros(N*N - agents, dtype = 'i, i')
        
            # insert agents according to their densities
            for i in range(N*N):
                if i < residents:
                    self.city[i] = self.resident_value
                elif residents <= i < residents + merchants:
                    self.city[i] = self.merchant_value
                else:
                    break
        
            # shuffle agents places
            np.random.shuffle(self.city)
            # reshape city array to matrix
            self.city = self.city.reshape((N,N))
            
        elif len(args) == 1 and isinstance(args[0], np.ndarray):
            self.city = args[0]
            N = self.city.shape[0]
            agents = 0
            residents = 0
            merchants = 0
            for i in range(N):
                for j in range(N):
                    if self.city[i][j] == self.resident_value:
                        agents += 1
                        residents += 1

                    elif self.city[i][j] == self.merchant_value:
                        agents += 1
                        merchants += 1
                        
            self.residentsArray = np.zeros(residents, dtype = 'i, i')
            self.merchantsArray = np.zeros(merchants, dtype = 'i, i')
            self.vacantArray = np.zeros(N*N - agents, dtype = 'i, i')

        else:
            print('Invalid input')
            print('Enter with a numpy LxL matrix or values for L, agent density and resident realtive density, in this order.')
            return None
        
        aux1 = 0
        aux2 = 0
        aux3 = 0
        for i in range(N):
            for j in range(N):
                if self.city[i][j] == self.resident_value:
                    self.residentsArray[aux1] = (i,j)
                    aux1 += 1

                elif self.city[i][j] == self.merchant_value:
                    self.merchantsArray[aux2] = (i,j)
                    aux2 += 1
                    
                else:
                    self.vacantArray[aux3] = (i,j)
                    aux3 += 1
                    

    def print_city(self):
        # creates a discrete colormap
        vacancy = np.array([147.0/255, 148.0/255, 150.0/255])  # grey
        agent =  np.array([255.0/255, 255.0/255, 0.0/255])     # red 
        cyan =  np.array([0.0/255, 200.0/255, 255.0/255])      # cyan
        cmap = colors.ListedColormap([cyan, vacancy, agent])
        # determines the limits of each color:
        bounds = [self.merchant_value, self.vacancy_value, self.resident_value, self.resident_value + 1]            
        norm= colors.BoundaryNorm(bounds, cmap.N)
    
        size = 8
        fig, ax = plt.subplots(figsize=(size,size))
        ax.imshow(self.city, cmap=cmap, norm=norm)
    
        # hide axis values
        plt.xticks([])  
        plt.yticks([])  
    
        plt.show()
        plt.close()


    def change_properties(self, args):
        """
        Receives as input a list of tuples, the first item is the variable name (string) that will be changed and the second item is its new value.
        """
        for arg in args:
            if arg[0] == 'neighborhood':
                self.neighborhood = arg[1]
                
            elif arg[0] == 'minimal_neighbors_resident_density':
                self.minimal_neighbors_resident_density = arg[1]

            elif arg[0] == 'maximal_neighbors_resident_density':
                self.maximal_neighbors_resident_density = arg[1]

            elif arg[0] == 'minimal_neighbors_merchant_density':
                self.minimal_neighbors_merchant_density = arg[1]

            elif arg[0] == 'minimal_merchant_neighbors_resident_density':
                self.minimal_merchant_neighbors_resident_density = arg[1]

            else:
                print(f'{arg[0]} is not a valid variable of the system.')

            # square neighborhood
            # 1:= 3x3; 2:= 5x5; 3:= 7x7; ...
            self.kernel_side = (2*self.neighborhood+1)
            self.max_neighbors = self.kernel_side**2 - 1

            # conditions of dissatisfaction Resident
            self.minimal_neighbors_resident = np.round( self.max_neighbors * self.minimal_neighbors_resident_density )
            self.maximal_neighbors_resident = np.round( self.max_neighbors * self.maximal_neighbors_resident_density )
            
            self.minimal_merchant_neighbors_resident = np.round(self.max_neighbors * self.minimal_merchant_neighbors_resident_density)
            self.maximal_merchant_neighbors_resident = self.max_neighbors
            
            # conditions of dissatisfaction Merchant
            self.minimal_neighbors_merchant = np.round( self.max_neighbors * self.minimal_neighbors_merchant_density )
            self.maximal_neighbors_merchant = self.max_neighbors
            
            self.minimal_resident_neighbors_merchant = 0
            self.maximal_resident_neighbors_merchant = self.max_neighbors
            
    
    def evolve(self, periodicBoundaries, timeSteps):
        for _ in range(timeSteps):
            if periodicBoundaries:
                boundary = 'wrap'
            else:
                boundary = 'fill'
                
            KERNEL = np.ones((self.kernel_side, self.kernel_side), dtype=np.int8)
            middle = int(self.kernel_side/2)
            KERNEL[middle][middle] = 0
            kws = dict(mode='same', boundary=boundary)
            Resident_neighs = convolve2d(self.city == self.resident_value,  KERNEL, **kws)
            Merchant_neighs = convolve2d(self.city == self.merchant_value,  KERNEL, **kws)
            Neighs = convolve2d(self.city != self.vacancy_value,  KERNEL, **kws)
        
            # conditions of dissatisfaction
            Resident_dissatisfied = ((((Neighs < self.minimal_neighbors_resident) | (Neighs > self.maximal_neighbors_resident)) |                                  # minimal and maximal neighbors                   
                                   ((Merchant_neighs < self.minimal_merchant_neighbors_resident) | (Merchant_neighs > self.maximal_merchant_neighbors_resident))) & # minimal and maximal merchant neighbors 
                                   (self.city == self.resident_value))

            Merchant_dissatisfied = ((((Neighs < self.minimal_neighbors_merchant) | (Neighs >self.maximal_neighbors_merchant)) |                                  # minimal and maximal neighbors   
                                   ((Resident_neighs < self.minimal_resident_neighbors_merchant) | (Resident_neighs > self.maximal_resident_neighbors_merchant))) & # minimal and maximal resident neighbors 
                                   (self.city == self.merchant_value))
            self.city[Resident_dissatisfied | Merchant_dissatisfied] = self.vacancy_value
            vacant = (self.city == self.vacancy_value).sum()
        
            N_Resident_dissatisfied, N_Merchant_dissatisfied = Resident_dissatisfied.sum(), Merchant_dissatisfied.sum()
            filling = np.full(vacant, self.vacancy_value, dtype=np.int8)
            filling[:N_Resident_dissatisfied] = self.resident_value
            filling[N_Resident_dissatisfied:N_Resident_dissatisfied + N_Merchant_dissatisfied] = self.merchant_value
            np.random.shuffle(filling)
            self.city[self.city == self.vacancy_value] = filling
    
        
    # unitary movement, each agent at a time
    def evolve_unitary(self, periodicBoundaries, timeSteps):
        for _ in range(timeSteps):
            # we choose an agent and a vancant site
            N = self.city.shape[0]
            rnd = np.random.random()
            residents = len(self.residentsArray)
            merchants = len(self.merchantsArray)
            agents = residents + merchants

            rndV = np.random.randint(0, len(self.vacantArray))    
            iV, jV = self.vacantArray[rndV]
        
            if rnd <= float(residents)/agents:
                rndA = np.random.randint(0, residents)
                iA, jA = self.residentsArray[rndA]

                neighs = 0 
                residentNeighs = 0
                merchantNeighs = 0
                for di in range(-self.neighborhood, self.neighborhood + 1):
                    for dj in range(-self.neighborhood, self.neighborhood + 1):
                        if di == 0 and dj == 0:
                            continue

                        if periodicBoundaries:
                            if self.city[(iA + di)%N][(jA+dj)%N] != self.vacancy_value:
                                neighs += 1
        
                                if self.city[(iA + di)%N][(jA+dj)%N] == self.resident_value:
                                    residentNeighs += 1
        
                                else:
                                    merchantNeighs += 1

                        else:
                            if 0 <= iA + di < N and 0 <= jA + dj < N: 
                                if self.city[iA + di][jA+dj] != self.vacancy_value:
                                    neighs += 1
            
                                    if self.city[(iA + di)%N][(jA+dj)%N] == self.resident_value:
                                        residentNeighs += 1
            
                                    else:
                                        merchantNeighs += 1
    
    
                agentDissatisfied = ((((neighs < self.minimal_neighbors_resident) | (neighs > self.maximal_neighbors_resident)) |                                  # minimal and maximal neighbors                   
                                    ((merchantNeighs < self.minimal_merchant_neighbors_resident) | (merchantNeighs > self.maximal_merchant_neighbors_resident)))) # minimal and maximal merchant neighbors

                if agentDissatisfied:
                    self.vacantArray[rndV] = (iA,jA)
                    self.residentsArray[rndA] = (iV,jV)
            
                    #update city grid
                    self.city[iA,jA] = self.vacancy_value
                    self.city[iV,jV] = self.resident_value
                
            else:
                rndA = np.random.randint(0, merchants)
                iA, jA = self.merchantsArray[rndA]
    
                neighs = 0 
                residentNeighs = 0
                merchantNeighs = 0
                for di in range(-self.neighborhood, self.neighborhood + 1):
                    for dj in range(-self.neighborhood, self.neighborhood + 1):
                        if di == 0 and dj ==0:
                            continue

                        if periodicBoundaries:
                            if self.city[(iA + di)%N][(jA+dj)%N] != self.vacancy_value:
                                neighs += 1
        
                                if self.city[(iA + di)%N][(jA+dj)%N] == self.resident_value:
                                    residentNeighs += 1
        
                                else:
                                    merchantNeighs += 1
                        else:
                            if 0 <= iA + di < N and 0 <= jA + dj < N: 
                                if self.city[(iA + di)%N][(jA+dj)%N] != self.vacancy_value:
                                    neighs += 1
            
                                    if self.city[(iA + di)%N][(jA+dj)%N] == self.resident_value:
                                        residentNeighs += 1
            
                                    else:
                                        merchantNeighs += 1
    
                agentDissatisfied = ((((neighs < self.minimal_neighbors_merchant) | (neighs > self.maximal_neighbors_merchant)) |                                  # minimal and maximal neighbors   
                                    ((residentNeighs < self.minimal_resident_neighbors_merchant) | (residentNeighs > self.maximal_resident_neighbors_merchant))))  # minimal and maximal resident neighbors


                if agentDissatisfied:
                    self.vacantArray[rndV] = (iA,jA)
                    self.merchantsArray[rndA] = (iV,jV)
                
                    #update city grid
                    self.city[iA,jA] = self.vacancy_value
                    self.city[iV,jV] = self.merchant_value

            #return neighs, residentNeighs, merchantNeighs, iA, jA, iV, jV


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
        
    def hk(self, target_value, periodicBoundaries):
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


        # PART OF THE CODE FOR PERIODIC CONDITIONS
        if periodicBoundaries:
            for j in range(m):
                if self.matrix[n-1][j] == target_value:
                    down = self.matrix[0][j]
    
                    if down == target_value:
                        self.union(self.labels[0][j], self.labels[n-1][j])
    
            for i in range(n):
                if self.matrix[i][m-1] == target_value:
                    right = self.matrix[i][0]
    
                    if right == target_value:
                        self.union(self.labels[i][0], self.labels[i][m-1])
    
            
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

            clustersSizesAverage = 0
            for label in clustersLabels:
                size = (self.labels == label).sum()
                clustersSizesAverage += size
                clustersSizes[label] = size

            clustersSizesAverage = float(clustersSizesAverage)/numClusters

            return numClusters, clustersSizes, clustersSizesAverage
            
        else:
            print('Run the algorithm first')    