import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from IPython import display
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


#################################################################################################################################################################################################################################


class City():

    merchant_value = -1
    vacancy_value = 0
    resident_value = 1

    #periodic boundaries
    periodicBoundaries = True
    
    # square neighborhood
    # 1:= 3x3; 2:= 5x5; 3:= 7x7; ...
    neighborhood = 1
    kernel_side = (2*neighborhood+1)
    max_neighbors = kernel_side**2 - 1
    
    # densities
    minimal_neighbors_resident_density = 0.25
    maximal_neighbors_resident_density = 0.8
    minimal_neighbors_merchant_density = 0.6
    maximal_neighbors_merchant_density = 1
    minimal_merchant_neighbors_resident_density = 0.2
    
    # conditions of dissatisfaction Resident
    minimal_neighbors_resident = minimal_neighbors_resident_density
    maximal_neighbors_resident = maximal_neighbors_resident_density
    
    minimal_merchant_neighbors_resident = minimal_merchant_neighbors_resident_density
    maximal_merchant_neighbors_resident = 1
    
    # conditions of dissatisfaction Merchant
    minimal_neighbors_merchant = minimal_neighbors_merchant_density
    maximal_neighbors_merchant = maximal_neighbors_merchant_density
    
    minimal_resident_neighbors_merchant = 0
    maximal_resident_neighbors_merchant = 1

    
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

        # receives as input a numpy matrix that represents the city    
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
    
        size = 10
        fig, ax = plt.subplots(figsize=(size,size))
        ax.imshow(self.city, cmap=cmap, norm=norm)
    
        # hide axis values
        plt.xticks([])  
        plt.yticks([])  
    
        plt.show()
        plt.close()


    def save_city_image(self, fileName):
        # creates a discrete colormap
        vacancy = np.array([147.0/255, 148.0/255, 150.0/255])  # grey
        agent =  np.array([255.0/255, 255.0/255, 0.0/255])     # red 
        cyan =  np.array([0.0/255, 200.0/255, 255.0/255])      # cyan
        cmap = colors.ListedColormap([cyan, vacancy, agent])
        # determines the limits of each color:
        bounds = [self.merchant_value, self.vacancy_value, self.resident_value, self.resident_value + 1]            
        norm= colors.BoundaryNorm(bounds, cmap.N)
    
        size = 10
        fig, ax = plt.subplots(figsize=(size,size))
        ax.imshow(self.city, cmap=cmap, norm=norm)
    
        # hide axis values
        plt.xticks([])  
        plt.yticks([])  

        path = 'steps/' + fileName + '.png'
    
        plt.savefig(path)
        plt.close()


    def change_properties(self, args):
        """
        Receives as input a list of tuples, the first item is the variable name (string) that will be changed and the second item is its new value.
        """
        for arg in args:
            if arg[0] == 'neighborhood':
                self.neighborhood = arg[1]

            elif arg[0] == 'periodicBoundaries':
                self.periodicBoundaries = arg[1]
                
            elif arg[0] == 'minimal_neighbors_resident_density':
                self.minimal_neighbors_resident_density = arg[1]

            elif arg[0] == 'maximal_neighbors_resident_density':
                self.maximal_neighbors_resident_density = arg[1]

            elif arg[0] == 'minimal_neighbors_merchant_density':
                self.minimal_neighbors_merchant_density = arg[1]

            elif arg[0] == 'maximal_neighbors_merchant_density':
                self.maximal_neighbors_merchant_density = arg[1]

            elif arg[0] == 'minimal_merchant_neighbors_resident_density':
                self.minimal_merchant_neighbors_resident_density = arg[1]

            else:
                print(f'{arg[0]} is not a valid variable of the system.')


            # square neighborhood
            # 1:= 3x3; 2:= 5x5; 3:= 7x7; ...
            self.kernel_side = (2*self.neighborhood+1)
            self.max_neighbors = self.kernel_side**2 - 1
            
            # conditions of dissatisfaction Resident
            self.minimal_neighbors_resident = self.minimal_neighbors_resident_density
            self.maximal_neighbors_resident = self. maximal_neighbors_resident_density
            
            self.minimal_merchant_neighbors_resident = self.minimal_merchant_neighbors_resident_density
            self.maximal_merchant_neighbors_resident = 1
            
            # conditions of dissatisfaction Merchant
            self.minimal_neighbors_merchant = self.minimal_neighbors_merchant_density
            self.maximal_neighbors_merchant = self.maximal_neighbors_merchant_density
            
            self.minimal_resident_neighbors_merchant = 0
            self.maximal_resident_neighbors_merchant = 1
            
    
    def evolve(self, timeSteps):
        for _ in range(timeSteps):
            if self.periodicBoundaries:
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
            Max_Neighs = convolve2d(self.city != np.inf, KERNEL, **kws)

            safetyMeasument = 0.2
        
            # conditions of dissatisfaction
            Resident_dissatisfied = ((((Neighs < Max_Neighs * self.minimal_neighbors_resident - safetyMeasument) | (Neighs > Max_Neighs * self.maximal_neighbors_resident + safetyMeasument)) |                                  # minimal and maximal neighbors                   
                                   ((Merchant_neighs < Max_Neighs * self.minimal_merchant_neighbors_resident  - safetyMeasument) | (Merchant_neighs > Max_Neighs * self.maximal_merchant_neighbors_resident  + safetyMeasument))) & # minimal and maximal merchant neighbors 
                                   (self.city == self.resident_value))

            Merchant_dissatisfied = ((((Neighs < Max_Neighs * self.minimal_neighbors_merchant - safetyMeasument) | (Neighs > Max_Neighs * self.maximal_neighbors_merchant + safetyMeasument)) |                                  # minimal and maximal neighbors   
                                   ((Resident_neighs < Max_Neighs * self.minimal_resident_neighbors_merchant - safetyMeasument) | (Resident_neighs > Max_Neighs * self.maximal_resident_neighbors_merchant + safetyMeasument))) & # minimal and maximal resident neighbors 
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
    def evolve_unitary(self, timeSteps):
        for _ in range(timeSteps):
            # we choose an agent and a vancant site
            N = self.city.shape[0]
            rnd = np.random.random()
            residents = len(self.residentsArray)
            merchants = len(self.merchantsArray)
            agents = residents + merchants

            rndV = np.random.randint(0, len(self.vacantArray))    
            iV, jV = self.vacantArray[rndV]

            safetyMeasument = 0.2
        
            if rnd <= float(residents)/agents:
                rndA = np.random.randint(0, residents)
                iA, jA = self.residentsArray[rndA]

                neighs = 0 
                residentNeighs = 0
                merchantNeighs = 0
                total_neighs = 0
                
                for di in range(-self.neighborhood, self.neighborhood + 1):
                    for dj in range(-self.neighborhood, self.neighborhood + 1):
                        if di == 0 and dj == 0:
                            continue

                        if self.periodicBoundaries:
                            total_neighs = self.max_neighbors
                            if self.city[(iA + di)%N][(jA+dj)%N] != self.vacancy_value:
                                neighs += 1
        
                                if self.city[(iA + di)%N][(jA+dj)%N] == self.resident_value:
                                    residentNeighs += 1
        
                                else:
                                    merchantNeighs += 1

                        else:
                            if 0 <= iA + di < N and 0 <= jA + dj < N: 
                                total_neighs += 1
                                
                                if self.city[iA + di][jA+dj] != self.vacancy_value:
                                    neighs += 1
            
                                    if self.city[(iA + di)%N][(jA+dj)%N] == self.resident_value:
                                        residentNeighs += 1
            
                                    else:
                                        merchantNeighs += 1
    
    
                agentDissatisfied = ((((neighs < total_neighs * self.minimal_neighbors_resident - safetyMeasument) | (neighs > total_neighs * self.maximal_neighbors_resident + safetyMeasument)) |                                  # minimal and maximal neighbors       
                                    ((merchantNeighs < total_neighs * self.minimal_merchant_neighbors_resident - safetyMeasument) | (merchantNeighs > total_neighs * self.maximal_merchant_neighbors_resident + safetyMeasument)))) # minimal and maximal merchant neighbors

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
                total_neighs = 0
                for di in range(-self.neighborhood, self.neighborhood + 1):
                    for dj in range(-self.neighborhood, self.neighborhood + 1):
                        if di == 0 and dj ==0:
                            continue

                        if self.periodicBoundaries:
                            total_neighs = self.max_neighbors
                            if self.city[(iA + di)%N][(jA+dj)%N] != self.vacancy_value:
                                neighs += 1
        
                                if self.city[(iA + di)%N][(jA+dj)%N] == self.resident_value:
                                    residentNeighs += 1
        
                                else:
                                    merchantNeighs += 1
                                    
                        else:
                            if 0 <= iA + di < N and 0 <= jA + dj < N: 
                                total_neighs += 1
                                if self.city[(iA + di)%N][(jA+dj)%N] != self.vacancy_value:
                                    neighs += 1
            
                                    if self.city[(iA + di)%N][(jA+dj)%N] == self.resident_value:
                                        residentNeighs += 1
            
                                    else:
                                        merchantNeighs += 1
    
                agentDissatisfied = ((((neighs < total_neighs * self.minimal_neighbors_merchant - safetyMeasument) | (neighs > total_neighs * self.maximal_neighbors_merchant + safetyMeasument)) |                                  # minimal and maximal neighbors   
                                    ((residentNeighs < total_neighs * self.minimal_resident_neighbors_merchant - safetyMeasument) | (residentNeighs > total_neighs * self.maximal_resident_neighbors_merchant + safetyMeasument))))  # minimal and maximal resident neighbors


                if agentDissatisfied:
                    self.vacantArray[rndV] = (iA,jA)
                    self.merchantsArray[rndA] = (iV,jV)
                
                    #update city grid
                    self.city[iA,jA] = self.vacancy_value
                    self.city[iV,jV] = self.merchant_value

            #return total_neighs, neighs, residentNeighs, merchantNeighs, iA, jA, iV, jV


    def simulate(self, timeSteps, unitaryEvolution, plot, plotInterval = 5, plotEnd = True, calculateEnergy = False, calculateDissatisfied = False):
        if calculateEnergy:
            energies = np.full(timeSteps, 1)

        if calculateDissatisfied:
            dissatisfieds = np.full(timeSteps, -1)

        # the unitary evolution energy and lyapunov function computations haven't been testet yet
        if unitaryEvolution:
            for i in range(timeSteps * 1000):
                self.evolve_unitary(1)

                if i%1000 == 0 and calculateEnergy:
                    energies[i//1000] = self.calculate_energy()

                if i%1000 == 0 and calculateDissatisfied:
                    dissatisfieds[i//1000] = self.calculate_dissatisfied()
            
                if i%(plotInterval*1000) == 0 and plot:
                    self.print_city()
                    display.clear_output(wait=True)

            if plotEnd:
                display.clear_output(wait=True)
                self.print_city()
            
        
        else:
            totalSatisfaction = False
            for i in range(timeSteps):
                # stop condition
                if i%10 == 0:
                    oldCity = np.copy(self.city)
                    self.evolve(1)
                    if calculateEnergy:
                        energies[i] = self.calculate_energy()

                    if calculateDissatisfied:
                        dissatisfieds[i] = self.calculate_dissatisfied()
                        
                    dif = oldCity - self.city
                    if all(all(element == 0) for element in dif):
                        if plot:
                            self.print_city()
                        totalSatisfaction = True
                        break
        
                else:
                    self.evolve(1)
                    if calculateEnergy:
                        energies[i] = self.calculate_energy()

                    if calculateDissatisfied:
                        dissatisfieds[i] = self.calculate_dissatisfied()
        
                if i%plotInterval == 0 and plot:
                    self.print_city()
                    display.clear_output(wait=True)

            if calculateEnergy and totalSatisfaction:
                energies = energies[:i]

            if calculateDissatisfied and totalSatisfaction:
                dissatisfieds = dissatisfieds[:i]

            if plotEnd:
                display.clear_output(wait=True)
                if totalSatisfaction:
                    print('ALL AGENTS ARE SATISFIED')
                else:
                    print('EXISTS AT LEAST ONE DISSATISFIED AGENT')
                self.print_city()

        if calculateEnergy and calculateDissatisfied:
            return energies, dissatisfieds

        if calculateEnergy:
            return energies

        if calculateDissatisfied:
            return dissatisfieds


    def simulate_and_create_gif(self, timeSteps, unitaryEvolution, plot, plotInterval = 5, plotEnd = True, imagesInGif = 15):
        stepsToSaveImage = np.linspace(0, timeSteps-1, imagesInGif, endpoint = True) 
        for i in range(len(stepsToSaveImage)):
            stepsToSaveImage[i] = int(np.round(stepsToSaveImage[i]))

        if unitaryEvolution:
            stepsToSaveImage = stepsToSaveImage*1000
            counter = -1
            for i in range(timeSteps * 1000):
                if i in stepsToSaveImage:
                    counter += 1
                    self.save_city_image('image_' + str(counter))
                
                self.evolve_unitary(1)
            
                if i%(plotInterval*1000) == 0 and plot:
                    self.print_city()
                    display.clear_output(wait=True)

            if plotEnd:
                display.clear_output(wait=True)
                self.print_city()
            
        
        else:
            counter = -1
            for i in range(timeSteps):
                if i in stepsToSaveImage:
                    counter += 1
                    self.save_city_image('image_' + str(counter))
                
                self.evolve(1)
        
                if i%plotInterval == 0 and plot:
                    self.print_city()
                    display.clear_output(wait=True)

            if plotEnd:
                display.clear_output(wait=True)
                self.print_city()


    def calculate_energy(self):
        """
        This function only calculates the merchant's energy
        """
        
        if self.periodicBoundaries:
            boundary = 'wrap'
        else:
            boundary = 'fill'
            
        KERNEL = np.ones((self.kernel_side, self.kernel_side), dtype=np.int8)
        middle = int(self.kernel_side/2)
        KERNEL[middle][middle] = 0
        kws = dict(mode='same', boundary=boundary)
        Neighs = convolve2d(self.city != self.vacancy_value,  KERNEL, **kws)
        isOccupied = (self.city != self.vacancy_value)
        
        neighborsOfAgents = Neighs * isOccupied
        energy = -0.5 * neighborsOfAgents.sum()
            
        return energy

    
    def calculate_Lyapunov_function(self):
        """
        This function only calculates the agents's lyapunov function, energy analogue
        """
        
        if self.periodicBoundaries:
            boundary = 'wrap'
        else:
            boundary = 'fill'
            
        KERNEL = np.ones((self.kernel_side, self.kernel_side), dtype=np.int8)
        middle = int(self.kernel_side/2)
        KERNEL[middle][middle] = 0
        kws = dict(mode='same', boundary=boundary)
        
        Neighs = convolve2d(self.city != self.vacancy_value,  KERNEL, **kws)
        Max_Neighs = convolve2d(self.city != np.inf, KERNEL, **kws)
        
        lyapunov = 0
        for i in range(self.city.shape[0]):
            for j in range(self.city.shape[0]):
                if self.city[i][j] == self.merchant_value:
                    lyapunov += Neighs[i][j] - Max_Neighs[i][j] * self.maximal_neighbors_merchant_density

                elif self.city[i][j] == self.resident_value:
                    lyapunov += Neighs[i][j]**2 - Neighs[i][j] * (Max_Neighs[i][j] * (self.maximal_neighbors_resident_density + self.minimal_neighbors_resident_density ) ) + Max_Neighs[i][j]**2 * self.maximal_neighbors_resident_density * self.minimal_neighbors_resident_density 
            
        return lyapunov


    def calculate_dissatisfied(self):
        if self.periodicBoundaries:
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
        Max_Neighs = convolve2d(self.city != np.inf, KERNEL, **kws)

        safetyMeasument = 0.2
    
        # conditions of dissatisfaction
        Resident_dissatisfied = ((((Neighs < Max_Neighs * self.minimal_neighbors_resident - safetyMeasument) | (Neighs > Max_Neighs * self.maximal_neighbors_resident + safetyMeasument)) |                                  # minimal and maximal neighbors                   
                               ((Merchant_neighs < Max_Neighs * self.minimal_merchant_neighbors_resident  - safetyMeasument) | (Merchant_neighs > Max_Neighs * self.maximal_merchant_neighbors_resident  + safetyMeasument))) & # minimal and maximal merchant neighbors 
                               (self.city == self.resident_value))

        Merchant_dissatisfied = ((((Neighs < Max_Neighs * self.minimal_neighbors_merchant - safetyMeasument) | (Neighs > Max_Neighs * self.maximal_neighbors_merchant + safetyMeasument)) |                                  # minimal and maximal neighbors   
                               ((Resident_neighs < Max_Neighs * self.minimal_resident_neighbors_merchant - safetyMeasument) | (Resident_neighs > Max_Neighs * self.maximal_resident_neighbors_merchant + safetyMeasument))) & # minimal and maximal resident neighbors 
                               (self.city == self.merchant_value))

        dissatified_matrix = Resident_dissatisfied | Merchant_dissatisfied
        dissatisfied = (dissatified_matrix == True).sum()
        
        return dissatisfied


#################################################################################################################################################################################################################################


class City_Continuous():

    merchant_value = -1
    vacancy_value = 0
    resident_value = 1

    #periodic boundaries
    periodicBoundaries = True
    
    # square neighborhood
    # 1:= 3x3; 2:= 5x5; 3:= 7x7; ...
    neighborhood = 1
    kernel_side = (2*neighborhood+1)
    max_neighbors = kernel_side**2 - 1
    
    # satisfaction functions
    # what is the reason to choose this resident satisfaction function (?). I couldn't find a teoretical reason
    def resident_satisfaction_function(self, num_neighs, max_neighbors):
        satisfaction = 1 - abs(1 - (2.0*num_neighs)/max_neighbors)

        return satisfaction
        
        
    def merchant_satisfaction_function(self, num_neighs, max_neighbors):
        satisfaction = (1.0*num_neighs)/max_neighbors

        return satisfaction

    
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

        # receives as input a numpy matrix that represents the city    
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
    
        size = 10
        fig, ax = plt.subplots(figsize=(size,size))
        ax.imshow(self.city, cmap=cmap, norm=norm)
    
        # hide axis values
        plt.xticks([])  
        plt.yticks([])  
    
        plt.show()
        plt.close()


    def save_city_image(self, fileName):
        # creates a discrete colormap
        vacancy = np.array([147.0/255, 148.0/255, 150.0/255])  # grey
        agent =  np.array([255.0/255, 255.0/255, 0.0/255])     # red 
        cyan =  np.array([0.0/255, 200.0/255, 255.0/255])      # cyan
        cmap = colors.ListedColormap([cyan, vacancy, agent])
        # determines the limits of each color:
        bounds = [self.merchant_value, self.vacancy_value, self.resident_value, self.resident_value + 1]            
        norm= colors.BoundaryNorm(bounds, cmap.N)
    
        size = 10
        fig, ax = plt.subplots(figsize=(size,size))
        ax.imshow(self.city, cmap=cmap, norm=norm)
    
        # hide axis values
        plt.xticks([])  
        plt.yticks([])  

        path = 'steps/' + fileName + '.png'
    
        plt.savefig(path)
        plt.close()


    def change_properties(self, args):
        """
        Receives as input a list of tuples, the first item is the variable name (string) that will be changed and the second item is its new value.
        """
        for arg in args:
            if arg[0] == 'neighborhood':
                self.neighborhood = arg[1]

            elif arg[0] == 'periodicBoundaries':
                self.periodicBoundaries = arg[1]
                
            else:
                print(f'{arg[0]} is not a valid variable of the system.')


            # square neighborhood
            # 1:= 3x3; 2:= 5x5; 3:= 7x7; ...
            self.kernel_side = (2*self.neighborhood+1)
            self.max_neighbors = self.kernel_side**2 - 1
            
    
    def evolve(self, timeSteps, calculateDissatisfied = False):

        if calculateDissatisfied:
            dissatisfieds = np.full(timeSteps, -1)
        
        for i in range(timeSteps):
            if self.periodicBoundaries:
                boundary = 'wrap'
            else:
                boundary = 'fill'

            L = self.city.shape[0]
            KERNEL = np.ones((self.kernel_side, self.kernel_side), dtype=np.int8)
            middle = int(self.kernel_side/2)
            KERNEL[middle][middle] = 0
            kws = dict(mode='same', boundary=boundary)
            Neighs = convolve2d(self.city != self.vacancy_value,  KERNEL, **kws)
            Neighs_of_residents = (self.city == self.resident_value) * Neighs
            Neighs_of_merchants = (self.city == self.merchant_value) * Neighs
            if not self.periodicBoundaries:
                Max_Neighs = convolve2d(self.city != np.inf, KERNEL, **kws)
                residents_satisfaction = self.resident_satisfaction_function(Neighs_of_residents, Max_Neighs)
                merchants_satisfaction = self.merchant_satisfaction_function(Neighs_of_merchants, Max_Neighs)
            else:
                residents_satisfaction = self.resident_satisfaction_function(Neighs_of_residents, self.max_neighbors)
                merchants_satisfaction = self.merchant_satisfaction_function(Neighs_of_merchants, self.max_neighbors)
            randomMatrix = np.random.rand(L,L)
        
            # conditions of dissatisfaction
            Resident_dissatisfied = ((residents_satisfaction <= randomMatrix) &  # agent satisfaction is 1 - probability to move
                                   (self.city == self.resident_value))

            Merchant_dissatisfied = ((merchants_satisfaction <= randomMatrix) &  # agent satisfaction is 1 - probability to move
                                   (self.city == self.merchant_value))

            """
            print(Neighs)
            print(Neighs_of_residents)
            print(Neighs_of_merchants)
            print(residents_satisfaction)
            print(merchants_satisfaction)
            print(randomMatrix)
            print(Resident_dissatisfied)
            print(Merchant_dissatisfied)
            """
            
            self.city[Resident_dissatisfied | Merchant_dissatisfied] = self.vacancy_value
            vacant = (self.city == self.vacancy_value).sum()
        
            N_Resident_dissatisfied, N_Merchant_dissatisfied = Resident_dissatisfied.sum(), Merchant_dissatisfied.sum()
            if calculateDissatisfied:
                dissatisfieds[i] = N_Resident_dissatisfied + N_Merchant_dissatisfied
            filling = np.full(vacant, self.vacancy_value, dtype=np.int8)
            filling[:N_Resident_dissatisfied] = self.resident_value
            filling[N_Resident_dissatisfied:N_Resident_dissatisfied + N_Merchant_dissatisfied] = self.merchant_value
            np.random.shuffle(filling)
            self.city[self.city == self.vacancy_value] = filling

        if calculateDissatisfied:
            return dissatisfieds
    

    # unitary movement, each agent at a time
    def evolve_unitary(self, timeSteps):
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
                total_neighs = 0
                
                for di in range(-self.neighborhood, self.neighborhood + 1):
                    for dj in range(-self.neighborhood, self.neighborhood + 1):
                        if di == 0 and dj == 0:
                            continue

                        if self.periodicBoundaries:
                            total_neighs = self.max_neighbors
                            if self.city[(iA + di)%N][(jA+dj)%N] != self.vacancy_value:
                                neighs += 1

                        else:
                            if 0 <= iA + di < N and 0 <= jA + dj < N: 
                                total_neighs += 1
                                
                                if self.city[iA + di][jA+dj] != self.vacancy_value:
                                    neighs += 1
    
                random = np.random.random()
                resident_satisfaction = self.resident_satisfaction_function(neighs, total_neighs)
                agentDissatisfied = (random >= resident_satisfaction)

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
                total_neighs = 0
                for di in range(-self.neighborhood, self.neighborhood + 1):
                    for dj in range(-self.neighborhood, self.neighborhood + 1):
                        if di == 0 and dj ==0:
                            continue

                        if self.periodicBoundaries:
                            total_neighs = self.max_neighbors
                            if self.city[(iA + di)%N][(jA+dj)%N] != self.vacancy_value:
                                neighs += 1
                                    
                        else:
                            if 0 <= iA + di < N and 0 <= jA + dj < N: 
                                total_neighs += 1
                                if self.city[(iA + di)%N][(jA+dj)%N] != self.vacancy_value:
                                    neighs += 1
    
                random = np.random.random()
                merchant_satisfaction = self.merchant_satisfaction_function(neighs, total_neighs)
                agentDissatisfied = (random >= merchant_satisfaction)

                if agentDissatisfied:
                    self.vacantArray[rndV] = (iA,jA)
                    self.merchantsArray[rndA] = (iV,jV)
                
                    #update city grid
                    self.city[iA,jA] = self.vacancy_value
                    self.city[iV,jV] = self.merchant_value

        #return total_neighs, neighs, iA, jA, iV, jV, random


    def simulate(self, timeSteps, unitaryEvolution, plot, plotInterval = 5, plotEnd = True, calculateEnergy = False, calculateDissatisfied = False):
        if calculateEnergy:
            energies = np.full(timeSteps, 1)

        if calculateDissatisfied:
            dissatisfieds = np.full(timeSteps, -1)

        if unitaryEvolution:
            for i in range(timeSteps * 1000):
                self.evolve_unitary(1)

                if i%1000 == 0 and calculateEnergy:
                    energies[i//1000] = self.calculate_energy()
            
                if i%(plotInterval*1000) == 0 and plot:
                    self.print_city()
                    display.clear_output(wait=True)

            if plotEnd:
                display.clear_output(wait=True)
                self.print_city()
                
        
        else:
            for i in range(timeSteps):
                if calculateDissatisfied:
                    dissatisfieds[i] = self.evolve(1, calculateDissatisfied)

                else: 
                    self.evolve(1)
                    
                if calculateEnergy:
                    energies[i] = self.calculate_energy()

                if i%plotInterval == 0 and plot:
                    self.print_city()
                    display.clear_output(wait=True)

            if plotEnd:
                display.clear_output(wait=True)
                self.print_city()

        if calculateEnergy and calculateDissatisfied:
            if unitaryEvolution:
                print("Dissatisfaction doesn't make that much sense whit unitary evolution, so it wasn't computed")
            return energies, dissatisfieds

        if calculateEnergy:
            return energies

        if calculateDissatisfied:
            if unitaryEvolution:
                print("Dissatisfaction doesn't make that much sense whit unitary evolution, so it wasn't computed")
            return dissatisfieds


    def simulate_and_create_gif(self, timeSteps, unitaryEvolution, plot, plotInterval = 5, plotEnd = True, imagesInGif = 15):
        stepsToSaveImage = np.linspace(0, timeSteps-1, imagesInGif, endpoint = True) 
        for i in range(len(stepsToSaveImage)):
            stepsToSaveImage[i] = int(np.round(stepsToSaveImage[i]))

        if unitaryEvolution:
            stepsToSaveImage = stepsToSaveImage*1000
            counter = -1
            for i in range(timeSteps * 1000):
                if i in stepsToSaveImage:
                    counter += 1
                    self.save_city_image('C_image_' + str(counter))
                
                self.evolve_unitary(1)
            
                if i%(plotInterval*1000) == 0 and plot:
                    self.print_city()
                    display.clear_output(wait=True)

            if plotEnd:
                display.clear_output(wait=True)
                self.print_city()
            
        
        else:
            counter = -1
            for i in range(timeSteps):
                if i in stepsToSaveImage:
                    counter += 1
                    self.save_city_image('C_image_' + str(counter))
                
                self.evolve(1)
        
                if i%plotInterval == 0 and plot:
                    self.print_city()
                    display.clear_output(wait=True)

            if plotEnd:
                display.clear_output(wait=True)
                self.print_city()


    def calculate_energy(self):
        """
        This function only calculates the merchant's energy
        """
        
        if self.periodicBoundaries:
            boundary = 'wrap'
        else:
            boundary = 'fill'
            
        KERNEL = np.ones((self.kernel_side, self.kernel_side), dtype=np.int8)
        middle = int(self.kernel_side/2)
        KERNEL[middle][middle] = 0
        kws = dict(mode='same', boundary=boundary)
        Neighs = convolve2d(self.city != self.vacancy_value,  KERNEL, **kws)
        isOccupied = (self.city != self.vacancy_value)
        
        neighborsOfAgents = Neighs * isOccupied
        energy = -0.5 * neighborsOfAgents.sum()
            
        return energy

    
    def calculate_Lyapunov_function(self):
        """
        This function only calculates the agents's lyapunov function, energy analogue
        """
        
        if self.periodicBoundaries:
            boundary = 'wrap'
        else:
            boundary = 'fill'
            
        KERNEL = np.ones((self.kernel_side, self.kernel_side), dtype=np.int8)
        middle = int(self.kernel_side/2)
        KERNEL[middle][middle] = 0
        kws = dict(mode='same', boundary=boundary)
        
        Neighs = convolve2d(self.city != self.vacancy_value,  KERNEL, **kws)
        Max_Neighs = convolve2d(self.city != np.inf, KERNEL, **kws)
        
        lyapunov = 0
        for i in range(self.city.shape[0]):
            for j in range(self.city.shape[0]):
                if self.city[i][j] == self.merchant_value:
                    lyapunov += Neighs[i][j] - Max_Neighs[i][j] * self.maximal_neighbors_merchant_density

                elif self.city[i][j] == self.resident_value:
                    lyapunov += Neighs[i][j]**2 - Neighs[i][j] * (Max_Neighs[i][j] * (self.maximal_neighbors_resident_density + self.minimal_neighbors_resident_density ) ) + Max_Neighs[i][j]**2 * self.maximal_neighbors_resident_density * self.minimal_neighbors_resident_density 
            
        return lyapunov


#################################################################################################################################################################################################################################


class City_Continuous_Logistic():

    merchant_value = -1
    vacancy_value = 0
    resident_value = 1

    #system temperature
    temperature = 0
    
    #periodic boundaries
    periodicBoundaries = True
    
    # square neighborhood
    # 1:= 3x3; 2:= 5x5; 3:= 7x7; ...
    neighborhood = 1
    kernel_side = (2*neighborhood+1)
    max_neighbors = kernel_side**2 - 1
    
    # satisfaction functions
    # what is the reason to choose this resident satisfaction function (?). I couldn't find a teoretical reason
    def resident_satisfaction_function(self, num_neighs, max_neighbors):
        satisfaction = 1 - abs(1 - (2.0*num_neighs)/max_neighbors)

        return satisfaction
        
        
    def merchant_satisfaction_function(self, num_neighs, max_neighbors):
        satisfaction = (1.0*num_neighs)/max_neighbors

        return satisfaction

    def movement_probability(self, difference):
        if self.temperature == 0:
            if difference > 0:
                prob = 1

            elif difference == 0:
                prob = 1/2

            else:
                prob = 0

        elif self.temperature < 0:
            print("Negative temperature is invalid")
            prob = 0

        else:
            prob = 1 / (1 + np.exp(-difference/self.temperature))

        return prob
        

    
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

        # receives as input a numpy matrix that represents the city    
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
    
        size = 10
        fig, ax = plt.subplots(figsize=(size,size))
        ax.imshow(self.city, cmap=cmap, norm=norm)
    
        # hide axis values
        plt.xticks([])  
        plt.yticks([])  
    
        plt.show()
        plt.close()


    def save_city_image(self, fileName):
        # creates a discrete colormap
        vacancy = np.array([147.0/255, 148.0/255, 150.0/255])  # grey
        agent =  np.array([255.0/255, 255.0/255, 0.0/255])     # red 
        cyan =  np.array([0.0/255, 200.0/255, 255.0/255])      # cyan
        cmap = colors.ListedColormap([cyan, vacancy, agent])
        # determines the limits of each color:
        bounds = [self.merchant_value, self.vacancy_value, self.resident_value, self.resident_value + 1]            
        norm= colors.BoundaryNorm(bounds, cmap.N)
    
        size = 10
        fig, ax = plt.subplots(figsize=(size,size))
        ax.imshow(self.city, cmap=cmap, norm=norm)
    
        # hide axis values
        plt.xticks([])  
        plt.yticks([])  

        path = 'steps/' + fileName + '.png'
    
        plt.savefig(path)
        plt.close()


    def change_properties(self, args):
        """
        Receives as input a list of tuples, the first item is the variable name (string) that will be changed and the second item is its new value.
        """
        for arg in args:
            if arg[0] == 'neighborhood':
                self.neighborhood = arg[1]

            elif arg[0] == 'periodicBoundaries':
                self.periodicBoundaries = arg[1]

            elif arg[0] == 'temperature':
                self.temperature = arg[1]
                
            else:
                print(f'{arg[0]} is not a valid variable of the system.')


            # square neighborhood
            # 1:= 3x3; 2:= 5x5; 3:= 7x7; ...
            self.kernel_side = (2*self.neighborhood+1)
            self.max_neighbors = self.kernel_side**2 - 1
            
    
    """
    def evolve(self, timeSteps, calculateDissatisfied = False):

        if calculateDissatisfied:
            dissatisfieds = np.full(timeSteps, -1)
        
        for i in range(timeSteps):
            if self.periodicBoundaries:
                boundary = 'wrap'
            else:
                boundary = 'fill'

            L = self.city.shape[0]
            KERNEL = np.ones((self.kernel_side, self.kernel_side), dtype=np.int8)
            middle = int(self.kernel_side/2)
            KERNEL[middle][middle] = 0
            kws = dict(mode='same', boundary=boundary)
            Neighs = convolve2d(self.city != self.vacancy_value,  KERNEL, **kws)
            Neighs_of_residents = (self.city == self.resident_value) * Neighs
            Neighs_of_merchants = (self.city == self.merchant_value) * Neighs
            if not self.periodicBoundaries:
                Max_Neighs = convolve2d(self.city != np.inf, KERNEL, **kws)
                residents_satisfaction = self.resident_satisfaction_function(Neighs_of_residents, Max_Neighs)
                merchants_satisfaction = self.merchant_satisfaction_function(Neighs_of_merchants, Max_Neighs)
            else:
                residents_satisfaction = self.resident_satisfaction_function(Neighs_of_residents, self.max_neighbors)
                merchants_satisfaction = self.merchant_satisfaction_function(Neighs_of_merchants, self.max_neighbors)
            randomMatrix = np.random.rand(L,L)
        
            # conditions of dissatisfaction
            Resident_dissatisfied = ((residents_satisfaction <= randomMatrix) &  # agent satisfaction is 1 - probability to move
                                   (self.city == self.resident_value))

            Merchant_dissatisfied = ((merchants_satisfaction <= randomMatrix) &  # agent satisfaction is 1 - probability to move
                                   (self.city == self.merchant_value))
            self.city[Resident_dissatisfied | Merchant_dissatisfied] = self.vacancy_value
            vacant = (self.city == self.vacancy_value).sum()
        
            N_Resident_dissatisfied, N_Merchant_dissatisfied = Resident_dissatisfied.sum(), Merchant_dissatisfied.sum()
            if calculateDissatisfied:
                dissatisfieds[i] = N_Resident_dissatisfied + N_Merchant_dissatisfied
            filling = np.full(vacant, self.vacancy_value, dtype=np.int8)
            filling[:N_Resident_dissatisfied] = self.resident_value
            filling[N_Resident_dissatisfied:N_Resident_dissatisfied + N_Merchant_dissatisfied] = self.merchant_value
            np.random.shuffle(filling)
            self.city[self.city == self.vacancy_value] = filling

        if calculateDissatisfied:
            return dissatisfieds
    """

    # unitary movement, each agent at a time
    def evolve_unitary(self, timeSteps):
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

                neighs_A = 0 
                total_neighs_A = 0
                neighs_V = 0
                total_neighs_V = 0
                
                for di in range(-self.neighborhood, self.neighborhood + 1):
                    for dj in range(-self.neighborhood, self.neighborhood + 1):
                        if di == 0 and dj == 0:
                            continue

                        if self.periodicBoundaries:
                            total_neighs_A = self.max_neighbors
                            total_neighs_V = self.max_neighbors
                            if self.city[(iA + di)%N][(jA+dj)%N] != self.vacancy_value:
                                neighs_A += 1

                            if self.city[(iV + di)%N][(jV+dj)%N] != self.vacancy_value:
                                neighs_V += 1

                        else:
                            if 0 <= iA + di < N and 0 <= jA + dj < N: 
                                total_neighs_A += 1
                                
                                if self.city[iA + di][jA+dj] != self.vacancy_value:
                                    neighs_A += 1

                            if 0 <= iV + di < N and 0 <= jV + dj < N: 
                                total_neighs_V += 1
                                
                                if self.city[iV + di][jV+dj] != self.vacancy_value:
                                    neighs_V += 1

                if self.periodicBoundaries:
                    if (abs(iA - iV) in [0, 1, self.city.shape[0]-1]) and (abs(jA - jV) in [0, 1, self.city.shape[0]-1]):
                        neighs_V -= 1

                else:
                    if (abs(iA - iV) <= 1) and (abs(jA - jV) <= 1):
                        neighs_V -= 1
            
                
                random = np.random.random()
                resident_satisfaction_A = self.resident_satisfaction_function(neighs_A, total_neighs_A)
                resident_satisfaction_V = self.resident_satisfaction_function(neighs_V, total_neighs_V)
                satisfaction_dif = resident_satisfaction_V - resident_satisfaction_A

                move_prob = self.movement_probability(satisfaction_dif)
                
                agenteMove = (random < move_prob)

                if agenteMove:
                    self.vacantArray[rndV] = (iA,jA)
                    self.residentsArray[rndA] = (iV,jV)
            
                    #update city grid
                    self.city[iA,jA] = self.vacancy_value
                    self.city[iV,jV] = self.resident_value
                
            else:
                rndA = np.random.randint(0, merchants)
                iA, jA = self.merchantsArray[rndA]
    
                neighs_A = 0 
                total_neighs_A = 0
                neighs_V = 0
                total_neighs_V = 0
                
                for di in range(-self.neighborhood, self.neighborhood + 1):
                    for dj in range(-self.neighborhood, self.neighborhood + 1):
                        if di == 0 and dj == 0:
                            continue

                        if self.periodicBoundaries:
                            total_neighs_A = self.max_neighbors
                            total_neighs_V = self.max_neighbors
                            if self.city[(iA + di)%N][(jA+dj)%N] != self.vacancy_value:
                                neighs_A += 1

                            if self.city[(iV + di)%N][(jV+dj)%N] != self.vacancy_value:
                                neighs_V += 1

                        else:
                            if 0 <= iA + di < N and 0 <= jA + dj < N: 
                                total_neighs_A += 1
                                
                                if self.city[iA + di][jA+dj] != self.vacancy_value:
                                    neighs_A += 1

                            if 0 <= iV + di < N and 0 <= jV + dj < N: 
                                total_neighs_V += 1
                                
                                if self.city[iV + di][jV+dj] != self.vacancy_value:
                                    neighs_V += 1

                if self.periodicBoundaries:
                    if (abs(iA - iV) in [0, 1, self.city.shape[0]-1]) and (abs(jA - jV) in [0, 1, self.city.shape[0]-1]):
                        neighs_V -= 1

                else:
                    if (abs(iA - iV) <= 1) and (abs(jA - jV) <= 1):
                        neighs_V -= 1
                        
                random = np.random.random()
                merchant_satisfaction_A = self.merchant_satisfaction_function(neighs_A, total_neighs_A)
                merchant_satisfaction_V = self.merchant_satisfaction_function(neighs_V, total_neighs_V)
                satisfaction_dif = merchant_satisfaction_V - merchant_satisfaction_A

                move_prob = self.movement_probability(satisfaction_dif)
                
                agenteMove = (random < move_prob)

                if agenteMove:
                    self.vacantArray[rndV] = (iA,jA)
                    self.merchantsArray[rndA] = (iV,jV)
                
                    #update city grid
                    self.city[iA,jA] = self.vacancy_value
                    self.city[iV,jV] = self.merchant_value


    def simulate(self, timeSteps, unitaryEvolution, plot, plotInterval = 5, plotEnd = True, calculateEnergy = False, calculateDissatisfied = False):
        if calculateEnergy:
            energies = np.full(timeSteps, 1)

        if calculateDissatisfied:
            dissatisfieds = np.full(timeSteps, -1)

        if unitaryEvolution:
            for i in range(timeSteps * 1000):
                self.evolve_unitary(1)

                if i%1000 == 0 and calculateEnergy:
                    energies[i//1000] = self.calculate_energy()
            
                if i%(plotInterval*1000) == 0 and plot:
                    self.print_city()
                    display.clear_output(wait=True)

            if plotEnd:
                display.clear_output(wait=True)
                self.print_city()
                
        
        else:
            print("Non-unitary evolution wasn't coded")

            """
            for i in range(timeSteps):
                if calculateDissatisfied:
                    dissatisfieds[i] = self.evolve(1, calculateDissatisfied)

                else: 
                    self.evolve(1)
                    
                if calculateEnergy:
                    energies[i] = self.calculate_energy()

                if i%plotInterval == 0 and plot:
                    self.print_city()
                    display.clear_output(wait=True)

            if plotEnd:
                display.clear_output(wait=True)
                self.print_city()

            """

        if calculateEnergy and calculateDissatisfied:
            if unitaryEvolution:
                print("Dissatisfaction doesn't make that much sense whit unitary evolution, so it wasn't computed")
            return energies, dissatisfieds

        if calculateEnergy:
            return energies

        if calculateDissatisfied:
            if unitaryEvolution:
                print("Dissatisfaction doesn't make that much sense whit unitary evolution, so it wasn't computed")
            return dissatisfieds


    def simulate_and_create_gif(self, timeSteps, unitaryEvolution, plot, plotInterval = 5, plotEnd = True, imagesInGif = 15):
        stepsToSaveImage = np.linspace(0, timeSteps-1, imagesInGif, endpoint = True) 
        for i in range(len(stepsToSaveImage)):
            stepsToSaveImage[i] = int(np.round(stepsToSaveImage[i]))

        if unitaryEvolution:
            stepsToSaveImage = stepsToSaveImage*1000
            counter = -1
            for i in range(timeSteps * 1000):
                if i in stepsToSaveImage:
                    counter += 1
                    self.save_city_image('CL_image_' + str(counter))
                
                self.evolve_unitary(1)
            
                if i%(plotInterval*1000) == 0 and plot:
                    self.print_city()
                    display.clear_output(wait=True)

            if plotEnd:
                display.clear_output(wait=True)
                self.print_city()
            
        
        else:
            print("Non-unitary evolution wasn't coded")

            """
            counter = -1
            for i in range(timeSteps):
                if i in stepsToSaveImage:
                    counter += 1
                    self.save_city_image('image_' + str(counter))
                
                self.evolve(1)
        
                if i%plotInterval == 0 and plot:
                    self.print_city()
                    display.clear_output(wait=True)

            if plotEnd:
                display.clear_output(wait=True)
                self.print_city()
            """


    def calculate_energy(self):
        """
        This function only calculates the merchant's energy
        """
        
        if self.periodicBoundaries:
            boundary = 'wrap'
        else:
            boundary = 'fill'
            
        KERNEL = np.ones((self.kernel_side, self.kernel_side), dtype=np.int8)
        middle = int(self.kernel_side/2)
        KERNEL[middle][middle] = 0
        kws = dict(mode='same', boundary=boundary)
        Neighs = convolve2d(self.city != self.vacancy_value,  KERNEL, **kws)
        isOccupied = (self.city != self.vacancy_value)
        
        neighborsOfAgents = Neighs * isOccupied
        energy = -0.5 * neighborsOfAgents.sum()
            
        return energy

    
    def calculate_Lyapunov_function(self):
        """
        This function only calculates the agents's lyapunov function, energy analogue
        """
        
        if self.periodicBoundaries:
            boundary = 'wrap'
        else:
            boundary = 'fill'
            
        KERNEL = np.ones((self.kernel_side, self.kernel_side), dtype=np.int8)
        middle = int(self.kernel_side/2)
        KERNEL[middle][middle] = 0
        kws = dict(mode='same', boundary=boundary)
        
        Neighs = convolve2d(self.city != self.vacancy_value,  KERNEL, **kws)
        Max_Neighs = convolve2d(self.city != np.inf, KERNEL, **kws)
        
        lyapunov = 0
        for i in range(self.city.shape[0]):
            for j in range(self.city.shape[0]):
                if self.city[i][j] == self.merchant_value:
                    lyapunov += Neighs[i][j] - Max_Neighs[i][j] * self.maximal_neighbors_merchant_density

                elif self.city[i][j] == self.resident_value:
                    lyapunov += Neighs[i][j]**2 - Neighs[i][j] * (Max_Neighs[i][j] * (self.maximal_neighbors_resident_density + self.minimal_neighbors_resident_density ) ) + Max_Neighs[i][j]**2 * self.maximal_neighbors_resident_density * self.minimal_neighbors_resident_density 
            
        return lyapunov


#################################################################################################################################################################################################################################


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


        # CODE FOR PERIODIC CONDITIONS
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