import numpy as np
import numba
from numba.experimental import jitclass
from numba import njit
import matplotlib.pyplot as plt
import random as rd

######################################################################################################################################################################################

"""
This class is used to store an agent's data. 

Every agent has a threshold and a state value, if the state value is 1 the agent is part of the riot, for exemple.
We are using the stochastic method (Model without sectors).

Using the sectors model, every Agent has a wish value, that stores the place that the Agent wants to go in the next time step,
and a sector value, that stores the place that the Agent currently is.

agent.sector == -1: it is in the reservoir; agent.sector == 0: it is in the sector_0; agent.sector == 1: it is in the sector_1. 
"""

spec2 = [
    ('threshold', numba.int64),               
    ('wish', numba.int64),
    ('sector', numba.int64),
    ('state', numba.int64),
    ('name', numba.int64),
]

@jitclass(spec2)
class Agent():
    
    def __init__(self, threshold):
        self.threshold = threshold
        self.wish = -1
        self.sector = -1
        self.state = 0
        self.name = -1
    
    
    def threshold_model(self, percentage):
        m = 0.2                                                                # if m -> inf, the model approaches the Granovetter's binary model of thresholds.
        probability = 1 / (1 + np.exp( m * (self.threshold - percentage) ) )   # stochastic model of thresholds [0 <= percentage <= 100]
        return probability
    
        
    # updates the state of an agent according with it's threshold, returns 1 if the agent enters the riot, for exemple, and returns 0 if nothing changes.     
    def update_state(self, percentage):              
        rnd = rd.random()
        if self.state == 0:
            if rnd <= self.threshold_model(percentage):
                self.state = 1
                return 1
            else:
                return 0
        else:
            return 0

        
    # updates the state of an agent according with it's threshold, returns 1 if the agent enters the riot, returns 0 if nothing chances and returns -1 if the agent exits the riot.
    def update_state_exit(self, percentage):  
        rnd = rd.random()
        if self.state == 0:
            if rnd <= self.threshold_model(percentage):
                self.state = 1
                return 1
            else:
                return 0
        else:
            if rnd > self.threshold_model(percentage):
                self.state = 0
                return -1
            else:
                return 0
            

######################################################################################################################################################################################            
"""
This class is used to create the simulation enviroment. 

"""
lista = numba.typed.List()
lista.append(Agent(10))
lista.clear()

spec1 = [
    ('reservoir', numba.typeof(lista)),               
    ('sector0', numba.typeof(lista)),
    ('sector0_size', numba.int64),
    ('sector1', numba.typeof(lista)),
    ('sector1_size', numba.int64),
]

@jitclass(spec1)
class System:
    
    # creates a system to be simulated. With one reservoir of Agents, and two riot sectors
    def __init__(self, agents, sector0_size, sector1_size):
        self.reservoir = agents
        self.sector0 = numba.typed.List([Agent(1)])
        self.sector0.clear()
        self.sector0_size = sector0_size
        self.sector1 = numba.typed.List([Agent(1)])
        self.sector1.clear()
        self.sector1_size = sector1_size
    
    # migrates an Agent "i" from the sector "sector"
    def migrate(self, sector, i):
        if sector == 0:
            agent = self.sector0[i]
            self.sector0.pop(i)
            self.sector1.append(agent)
            
        else:
            agent = self.sector1[i]
            self.sector1.pop(i)
            self.sector0.append(agent)
     
    
    # checks which Agent, in the reservoir, wants to riot according to it's threshold. It can go in sector 0 or 1.         
    def update_wishes_reservoir(self):
        for i in range(len(self.reservoir)):
            num = rd.randint(0,1)
            rnd = rd.random()
            if num == 0:
                if len(self.sector0) < self.sector0_size:
                    percentage = len(self.sector0)/self.sector0_size * 100
                    if rnd <= self.reservoir[i].threshold_model(percentage):
                        self.reservoir[i].wish = 0
                        continue
                
                if len(self.sector1) < self.sector1_size:                    
                    percentage = len(self.sector1)/self.sector1_size * 100
                    if rnd <= self.reservoir[i].threshold_model(percentage):
                        self.reservoir[i].wish = 1
                        continue
                    
            else:
                if len(self.sector1) < self.sector1_size:                    
                    percentage = len(self.sector1)/self.sector1_size * 100
                    if rnd <= self.reservoir[i].threshold_model(percentage):
                        self.reservoir[i].wish = 1
                        continue
                    
                if len(self.sector0) < self.sector0_size:
                    percentage = len(self.sector0)/self.sector0_size * 100
                    if rnd <= self.reservoir[i].threshold_model(percentage):
                        self.reservoir[i].wish = 0
                        continue
    
    
    # checks which Agent, in both sectros, wants to exit the riot.              
    def update_wishes_sectors_exit(self):
        for i in range(len(self.sector0)):
            rnd = rd.random()
            percentage = len(self.sector0)/self.sector0_size * 100
            if rnd > self.sector0[i].threshold_model(percentage):
                self.sector0[i].wish = -1
                
        for i in range(len(self.sector1)):
            rnd = rd.random()
            percentage = len(self.sector1)/self.sector1_size * 100
            if rnd > self.sector1[i].threshold_model(percentage):
                self.sector1[i].wish = -1
     
    
    # checks which Agent, on both sectors, wants to migrate to other sector (random model)
    def update_wishes_sectors_migration_random(self,migration_probability):
        for i in range(len(self.sector0)):
            rnd = rd.random()
            if rnd <= migration_probability:
                self.sector0[i].wish = 1
                
        for i in range(len(self.sector1)):
            rnd = rd.random()
            if rnd <= migration_probability:
                self.sector1[i].wish = 0
                
                
    # checks which Agent, on both sectors, wants to migrate to other sector (random and unidirectional model)            
    def update_wishes_sectors_migration_random_unidirectional(self, migration_probability):
        for i in range(len(self.sector0)):
            rnd = rd.random()
            if rnd <= migration_probability:
                self.sector0[i].wish = 1
                
                
    # checks which Agent, on both sectors, wants to migrate to other sector (gregarious model)             
    def update_wishes_sectors_migration_gregarious(self):
        m = 10
        dif = len(self.sector1) - len(self.sector0)
        probability = 1 - np.exp(-m * dif)
        
        for i in range(len(self.sector0)):
            rnd = rd.random()
            if dif > 0 and rnd <= probability:
                self.sector0[i].wish = 1
                
        dif = len(self.sector0) - len(self.sector1)
        probability = 1 - np.exp(-m * dif)
        
        for i in range(len(self.sector1)):
            rnd = rd.random()
            if dif > 0 and rnd <= probability:
                self.sector1[i].wish = 0
    
    
    # moves the Agents from the reservoir to the sectors of their wishes values
    def update_reservoir(self):
        i = 0
        while i < len(self.reservoir):
            if self.reservoir[i].wish == 0 and len(self.sector0) < self.sector0_size:
                agent = self.reservoir[i]
                self.reservoir.pop(i)
                self.sector0.append(agent)
                agent.sector = agent.wish
                i -= 1
            elif self.reservoir[i].wish == 1 and len(self.sector1) < self.sector1_size:
                agent = self.reservoir[i]
                self.reservoir.pop(i)
                self.sector1.append(agent)
                agent.sector = agent.wish
                i -= 1
            else:
                self.reservoir[i].wish = self.reservoir[i].sector
            i += 1
    
    
    # moves the Agents from the sectors to the reservoir or to the other sector
    def update_sectors(self):
        i = 0
        while i < len(self.sector0):
            if self.sector0[i].wish == -1:
                agent = self.sector0[i]
                self.sector0.pop(i)
                self.reservoir.append(agent)
                agent.sector = agent.wish
                i -= 1
            elif self.sector0[i].wish == 1 and len(self.sector1) < self.sector1_size:
                agent = self.sector0[i]
                self.sector0.pop(i)
                self.sector1.append(agent)
                agent.sector = agent.wish
                i -= 1
            else:
                self.sector0[i].wish = self.sector0[i].sector
            i += 1
            
        i = 0
        while i < len(self.sector1):
            if self.sector1[i].wish == -1:
                agent = self.sector1[i]
                self.sector1.pop(i)
                self.reservoir.append(agent)
                agent.sector = agent.wish
                i -= 1
            elif self.sector1[i].wish == 0 and len(self.sector0) < self.sector0_size:
                agent = self.sector1[i]
                self.sector1.pop(i)
                self.sector0.append(agent)
                agent.sector = agent.wish
                i -= 1
            else:
                self.sector1[i].wish = self.sector1[i].sector
            i += 1
                
            
######################################################################################################################################################################################

"""
Generates an sample of thresholds according with a normal distribution of given average and standard deviation
"""

@njit
def create_thresholds(N = 100, average = 25, deviation = 10):
    """
    Inputs:
        Sample parameters:
          N := total number of agents
          average := average value of the normal distribution
          deviation := standard deviation of the normal distribution
    
    This function creates an array with N threshold values (0 <= x <= 100) according with a normal distribution with the given parameters.
          
    Outputs:      
        A np.array with the sorted thresholds values
    
    """
    
    thresholds = np.zeros(N)
    
    # generating the values
    for i in range(N):
        threshold = rd.gauss(average, deviation)     # generates a random value according with a normal distribution

        if threshold < 0:
            threshold = 0
        elif threshold > 100:
            threshold = 100

        thresholds[i] = threshold
        
    #thresholds = sorted(thresholds) #sorts the array
    
    return thresholds

######################################################################################################################################################################################

@njit
def create_agents(N = 100, average = 25, deviation = 10):
    """
    Inputs:
        N := total number of agents
        average := average value of the normal distribution
        deviation := standard deviation of the normal distribution
        
        The function creates an array of Agents according with the normal distribution of threshold values.
        
    Outputs:
        agents := array with all Agents
    
    """
    
    agents = numba.typed.List()
    
    thresholds = create_thresholds(N, average, deviation)
    
    for i in range(N):
        agents.append(Agent(thresholds[i]))
        
    return agents


######################################################################################################################################################################################

@njit
def name_agents(agents):
    """
    Inputs:
        agents := Agents array
        
        The function names each Agent in the agents array.
        
    Outputs:
        agents := array with all Agents
    
    """
    
    N = len(agents)
    
    for i in range(N):
        agents[i].name = i
        
    return agents


######################################################################################################################################################################################

"""
Riot simulation
"""

@njit
def simulate_riot(thresholds):
    """
    Inputs:
        thresholds := array with the Agents thresholds 
    
    This function calculates the size of an riot according with the Granovetter binary model: The Agent enters the riot if its threshold value is less or equal
    to the number (or percentage) of people rioting, otherwise, it doesn't.
    
    Outputs:
        A "answer" np.array with the riot's evolution over time
    
    """
    
    riot_size = 0
    progression = np.zeros(len(thresholds)+1, dtype = np.int64)    # array that stores the riot's evolution over time
    aux = 0
    count = 0
    
    while True:
        for i in range(len(thresholds)):
            if thresholds[i] <= riot_size: # check all Agents and counts which of them enters the riot
                aux += 1
        count += 1
        progression[count] = aux
        if riot_size == aux:   # if the riot size doesn't change in a time step it is stable
            break
        riot_size = aux
        aux = 0
        
    return progression[0:count]  # trocar esse count pelo i (dá na mesma(?))


######################################################################################################################################################################################


@njit
def simulate_riot_stochastic(agents, steps = 100):
    """
    Inputs:
        agents := Agents array
        steps := number of the simulation's time steps
    
    This function calculates the size of an riot according with the stochastic threshold model: The Agent has a higher probability of entering the riot if its threshold value 
    is less or equal to the number (or percentage) of people rioting, and has a low probability of entering the riot if its threshold value is less than the number (or percentage)
    of people rioting.
    
    Outputs:
         A "answer" np.array with the riot's evolution over time
    
    """
    
    riot_size = 0
    progression = np.zeros(steps+1)              # array that stores the riot's evolution over time
    
    for i in range(1,steps+1):
        for agent in agents:
            riot_size += agent.update_state(riot_size)
            
        progression[i] = riot_size
        
    return progression


######################################################################################################################################################################################


@njit
def simulate_riot_stochastic_2(agents, steps = 100):
    """
    Inputs:
        agents := Agents array
        steps := number of the simulation's time steps
    
    This function calculates the size of an riot according with the stochastic threshold model: The Agent has a higher probability of entering the riot if its threshold value 
    is less or equal to the number (or percentage) of people rioting, and has a low probability of entering the riot if its threshold value is less than the number (or percentage)
    of people rioting. But if, in a time step, nothing changes the simulation stops.
    
    Outputs:
        A "answer" np.array with the riot's evolution over time
    
    """
    
    riot_size = 0
    progression = np.zeros(steps+1)              # array that stores the riot's evolution over time
    aux = 0
    
    for i in range(1,steps+1):
        for agent in agents:
            aux += agent.update_state(riot_size)
        
        
        if aux == 0:
            break
            
        riot_size += aux            
        progression[i] = riot_size
        aux = 0
        
    return progression[:i]


######################################################################################################################################################################################


@njit
def simulate_riot_stochastic_exit(agents, steps = 100):
    """
    Inputs:
        agents := Agents array
        steps := number of the simulation's time steps
    
    This function calculates the size of an riot according with the stochastic threshold model: The Agent has a higher probability of entering the riot if its threshold value is less or equal to the number
    (or percentage) of people rioting, and has a low probability of entering the riot if its threshold value is less than the number (or percentage) of people rioting. Moreover, there is a chance of an Agent
    to exit the riot according to a logistic function.
    
    Outputs:
        A np.array with the sorted thresholds values
    
    """
    
    riot_size = 0
    progression = np.zeros(steps+1)                                     # array that stores the riot's evolution over time
    
    for i in range(1,steps+1):
        for agent in agents:
            riot_size += agent.update_state_exit(riot_size/len(agents) * 100)

        progression[i] = riot_size
        
    return progression


######################################################################################################################################################################################


@njit
def simulate_riot_stochastic_exit_intermediary(agents, steps = 100):
    """
    Inputs:
        agents := Agents array
        steps := number of the simulation's time steps
    
    This function calculates the size of an riot according with the stochastic threshold model: The Agent has a higher probability of entering the riot if its threshold value 
    is less or equal to the number (or percentage) of people rioting, and has a low probability of entering the riot if its threshold value is less than the number (or percentage)
    of people rioting.
    
    Outputs:
         A "answer" np.array with the riot's evolution over time
    
    """
    
    riot_size = 0
    states = np.full((steps+1,len(agents)),-1)
    progression = np.zeros(steps+1)              # array that stores the riot's evolution over time
    
    for i in range(1,steps+1):
        for j in range(len(agents)):
            riot_size += agents[j].update_state_exit(riot_size/len(agents) * 100)
            if agents[j].state == 1:
                states[i][j] = agents[j].threshold 
            
        progression[i] = riot_size
        
    return progression, states


######################################################################################################################################################################################


@numba.njit
def simulate_riot_sectors(system, steps = 50):
    """
    Inputs:
        system := System class variable that contains all Agents
        steps := number of the simulation's time steps
    
    This functions simulates a set of 2 simultaneous riots that occur in 2 distinct sectors using the stochastic threshold model. There are a set of Agents in a reservoir that can
    enter sectors 0 or 1 to riot. Each sector has a size, so the thrsehold of each Agent is based on the number of Agents rioting in a sector compared with the number of Agents that
    can be in that sector.
     
    
    Outpurs:
         A np.array "progression" that contains the time evolution of each riots over time.
         
    """

    progression = np.zeros((2,steps+1), dtype = np.int64)              # array that stores the riot's evolution over time
    
    for i in range(1,steps+1):
        system.update_wishes_reservoir()             # check reservoir Agents (enter riot)
        system.update_reservoir()                    # move Agents from reservoir
           
        progression[0][i] = len(system.sector0)
        progression[1][i] = len(system.sector1)
            
    return progression


######################################################################################################################################################################################


@numba.njit
def simulate_riot_sectors_exit(system, steps = 50):
    """
    Inputs:
        system := System class variable that contains all Agents
        steps := number of the simulation's time steps
    
    This functions simulates a set of 2 simultaneous riots that occur in 2 distinct sectors using the stochastic threshold model. There are a set of Agents in a reservoir that can
    enter sectors 0 or 1 to riot. Each sector has a size, so the thrsehold of each Agent is based on the number of Agents rioting in a sector compared with the number of Agents that
    can be in that sector. In this function, the Agents can exit the riot and return to the reservoir according to a logistic function.
     
    
    Outpurs:
         A np.array "progression" that contains the time evolution of each riots over time.
         
    """
    
    progression = np.zeros((2,steps+1))              # array that stores the riot's evolution over time
    
    for i in range(1,steps+1):
        system.update_wishes_reservoir()             # check reservoir Agents (enter riot)
        system.update_wishes_sectors_exit()          # check sectors Agents (leave riot)
        system.update_reservoir()                    # move Agents from reservoir
        system.update_sectors()                      # move Agents from sectors
           
        progression[0][i] = len(system.sector0)
        progression[1][i] = len(system.sector1)
            
    return progression


######################################################################################################################################################################################


@numba.njit
def simulate_riot_sectors_migration(system, steps = 50, migration_probability = 0.01):
    """
    Inputs:
        system := System class variable that contains all Agents
        steps := number of the simulation's time steps
    
    This functions simulates a set of 2 simultaneous riots that occur in 2 distinct sectors using the stochastic threshold model. There are a set of Agents in a reservoir that can
    enter sectors 0 or 1 to riot. Each sector has a size, so the thrsehold of each Agent is based on the number of Agents rioting in a sector compared with the number of Agents that
    can be in that sector. In this function, the Agents in a section can migrate to the other sector with probability equals to "migration_probability".
     
    
    Outpurs:
         A np.array "progression" that contains the time evolution of each riots over time.
         
    """
    
    progression = np.zeros((2,steps+1))              # array that stores the riot's evolution over time
    
    for i in range(1,steps+1):
        system.update_wishes_reservoir()                                     # check reservoir Agents (enter riot)
        system.update_wishes_sectors_migration_random(migration_probability) # check sectors Agents (migrate to other sector)
        system.update_reservoir()                                            # move Agents from reservoir
        system.update_sectors()                                              # move Agents from sectors
           
        progression[0][i] = len(system.sector0)
        progression[1][i] = len(system.sector1)
            
    return progression


######################################################################################################################################################################################


@numba.njit
def simulate_riot_sectors_migration_exit(system, steps = 50, migration_probability = 0.01, start = 0):
    """
    Inputs:
        system := System class variable that contains all Agents
        steps := number of the simulation's time steps
        migration_probability := the probability of an Agent migrate in a time step
        start := the time step value when the Agents can migrate between sectors
    
    This functions simulates a set of 2 simultaneous riots that occur in 2 distinct sectors using the stochastic threshold model. There are a set of Agents in a reservoir that can
    enter sectors 0 or 1 to riot. Each sector has a size, so the thrsehold of each Agent is based on the number of Agents rioting in a sector compared with the number of Agents that
    can be in that sector. In this function, the Agents in a section can migrate to the other sector with probability equals to "migration_probability" if "i", the current time step is
    greater or equal to "start". Moreover, the Agents can exit the riot and return to the reservoir according to a logistic function.
     
    
    Outpurs:
         A np.array "progression" that contains the time evolution of each riots over time.
        
    """
    
    progression = np.zeros((2,steps+1))              # array that stores the riot's evolution over time
    
    for i in range(1,steps+1):
        system.update_wishes_reservoir()                                         # check reservoir Agents (enter riot)
        system.update_wishes_sectors_exit()                                      # check sectors Agents (leave riot)
        if i >= start:
            system.update_wishes_sectors_migration_random(migration_probability) # check sectors Agents (migrate to other sector)
        system.update_reservoir()                                                # move Agents from reservoir
        system.update_sectors()                                                  # move Agents from sectors
           
        progression[0][i] = len(system.sector0)
        progression[1][i] = len(system.sector1)
            
    return progression


######################################################################################################################################################################################


@numba.njit
def simulate_riot_sectors_migration_gregarious(system, steps = 50, start = 0):
    """
    Inputs:
        system := System class variable that contains all Agents
        steps := number of the simulation's time steps
        start := the time step value when the Agents can migrate between sectors
    
    This functions simulates a set of 2 simultaneous riots that occur in 2 distinct sectors using the stochastic threshold model. There are a set of Agents in a reservoir that can
    enter sectors 0 or 1 to riot. Each sector has a size, so the thrsehold of each Agent is based on the number of Agents rioting in a sector compared with the number of Agents that
    can be in that sector. In this function, the Agents in a section can migrate to the other sector according to the difference between the number of Agents in each sector if "i",
    the current time step is greater or equal to "start".
     
    
    Outpurs:
         A np.array "progression" that contains the time evolution of each riots over time.
        
    """
    
    progression = np.zeros((2,steps+1))              # array that stores the riot's evolution over time
    
    for i in range(1,steps+1):
        system.update_wishes_reservoir()                                             # check reservoir Agents (enter riot)
        if i >= start:
            system.update_wishes_sectors_migration_gregarious() # check sectors Agents (migrate to other sector)
        system.update_reservoir()                                                    # move Agents from reservoir
        system.update_sectors()                                                      # move Agents from sectors
           
        progression[0][i] = len(system.sector0)
        progression[1][i] = len(system.sector1)
            
    return progression


######################################################################################################################################################################################


@numba.njit
def simulate_riot_sectors_migration_gregarious_exit(system, steps = 50, start = 0):
    """
    Inputs:
        system := System class variable that contains all Agents
        steps := number of the simulation's time steps
        start := the time step value when the Agents can migrate between sectors
    
    This functions simulates a set of 2 simultaneous riots that occur in 2 distinct sectors using the stochastic threshold model. There are a set of Agents in a reservoir that can
    enter sectors 0 or 1 to riot. Each sector has a size, so the thrsehold of each Agent is based on the number of Agents rioting in a sector compared with the number of Agents that
    can be in that sector. In this function, the Agents in a section can migrate to the other sector according to the difference between the number of Agents in each sector if "i",
    the current time step is greater or equal to "start". Moreover, the Agents can exit the riot and return to the reservoir according to a logistic function.
     
    
    Outpurs:
         A np.array "progression" that contains the time evolution of each riots over time.
        
    """
    
    progression = np.zeros((2,steps+1))              # array that stores the riot's evolution over time
    
    for i in range(1,steps+1):
        system.update_wishes_reservoir()                                             # check reservoir Agents (enter riot)
        system.update_wishes_sectors_exit()                                          # check sectors Agents (leave riot)
        if i >= start:
            system.update_wishes_sectors_migration_gregarious() # check sectors Agents (migrate to other sector)
        system.update_reservoir()                                                    # move Agents from reservoir
        system.update_sectors()                                                      # move Agents from sectors
           
        progression[0][i] = len(system.sector0)
        progression[1][i] = len(system.sector1)
            
    return progression


######################################################################################################################################################################################


@numba.njit
def simulate_riot_sectors_migration_exit_unidirectional(system, steps = 50, migration_probability = 0.01, start = 0):
    """
    Inputs:
        system := System class variable that contains all Agents
        steps := number of the simulation's time steps
        migration_probability := the probability of an Agent migrate in a time step
        start := the time step value when the Agents can migrate between sectors
    
    This functions simulates a set of 2 simultaneous riots that occur in 2 distinct sectors using the stochastic threshold model. There are a set of Agents in a reservoir that can
    enter sectors 0 or 1 to riot. Each sector has a size, so the thrsehold of each Agent is based on the number of Agents rioting in a sector compared with the number of Agents that
    can be in that sector. In this function, the Agents in a section can migrate from sector 0 to sector 1 with probability equals to "migration_probability" if "i", the current time step is
    greater or equal to "start". Moreover, the Agents can exit the riot and return to the reservoir according to a logistic function.
     
    
    Outpurs:
         A np.array "progression" that contains the time evolution of each riots over time.
        
    """
    
    progression = np.zeros((2,steps+1))              # array that stores the riot's evolution over time
    
    for i in range(1,steps+1):
        system.update_wishes_reservoir()                                                        # check reservoir Agents (enter riot)
        system.update_wishes_sectors_exit()                                                     # check sectors Agents (leave riot)
        if i >= start:
            system.update_wishes_sectors_migration_random_unidirectional(migration_probability) # check sector0 Agents (migrate to other sector)
        system.update_reservoir()                                                               # move Agents from reservoir
        system.update_sectors()                                                                 # move Agents from sectors
           
        progression[0][i] = len(system.sector0)
        progression[1][i] = len(system.sector1)
            
    return progression