import numpy as np
import matplotlib.pyplot as plt
import random as rd

######################################################################################################################################################################################

"""
This class is used to create the simulation enviroment. 

"""

class System:
    
    
    # creates a system to be simulated. With one reservoir of Agents, and two riot sectors
    def __init__(self, agents, limit_0, limit_1):
        self.reservoir = agents
        self.sector0 = np.array([])
        self.sector0_size = limit_0
        self.sector1 = np.array([])
        self.sector1_size = limit_1
     
    
    # migrates an Agent "i" from the sector "sector"
    def migrate(self, sector, i):
        if sector == 0:
            agent = self.sector0[i]
            self.sector0 = np.delete(self.sector0, i)
            self.sector1 = np.append(self.sector1, agent)
            
        else:
            agent = self.sector1[i]
            self.sector1 = np.delete(self.sector1, i)
            self.sector0 = np.append(self.sector0, agent)
     
    
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
    def update_wishes_sectors(self):
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
     
    
    def update_wishes_migration_random(self,migration_probability):
        for i in range(len(self.sector0)):
            rnd = rd.random()
            if rnd > migration_probability:
                self.sector0[i].wish = 1
                
        for i in range(len(self.sector1)):
            rnd = rd.random()
            if rnd > migration_probability:
                self.sector1[i].wish = 0
    
    
    # moves the Agents from the reservoir to the sectors of their wishes values
    def update_reservoir(self):
        i = 0
        while i < len(self.reservoir):
            if self.reservoir[i].wish == 0 and len(self.sector0) < self.sector0_size:
                agent = self.reservoir[i]
                self.reservoir = np.delete(self.reservoir, i)
                self.sector0 = np.append(self.sector0, agent)
                i -= 1
            elif self.reservoir[i].wish == 1 and len(self.sector1) < self.sector1_size:
                agent = self.reservoir[i]
                self.reservoir = np.delete(self.reservoir, i)
                self.sector1 = np.append(self.sector1, agent)
                i -= 1
            i += 1
    
    
    # moves the Agents from the sectors to the reservoir or to the other sector
    def update_sectors(self):
        i = 0
        while i < len(self.sector0):
            if self.sector0[i].wish == -1:
                agent = self.sector0[i]
                self.sector0 = np.delete(self.sector0, i)
                self.reservoir = np.append(self.reservoir, agent)
                i -= 1
            elif self.sector0[i].wish == 1 and len(self.sector1) < self.sector1_size:
                agent = self.sector0[i]
                self.sector0 = np.delete(self.sector0, i)
                self.sector1 = np.append(self.sector1, agent)
                i -= 1
            i += 1
            
        i = 0
        while i < len(self.sector1):
            if self.sector1[i].wish == -1:
                agent = self.sector1[i]
                self.sector1 = np.delete(self.sector1, i)
                self.reservoir = np.append(self.reservoir, agent)
                i -= 1
            elif self.sector1[i].wish == 0 and len(self.sector0) < self.sector0_size:
                agent = self.sector1[i]
                self.sector0 = np.delete(self.sector1, i)
                self.sector1 = np.append(self.sector0, agent)
                i -= 1
            i += 1
                
            
######################################################################################################################################################################################

"""
This class is used to store an agent's data. 

Every agent has a threshold and a state value, if the state value is 1 the agent is part of the riot, for exemple.
We are using the stochastic method.

agent.sector == -1: it is in the reservoir; agent.sector == 0: it is in the sector_0; agent.sector == 1: it is in the sector_1. 
"""


class Agent:
    
    
    def __init__(self, threshold):
        self.threshold = threshold
        self.wish = -1
        self.sector = -1
    
    
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
Generates an sample of thresholds according with a normal distribution of given average and standard deviation
"""


def create_thresholds(N = 100, a = True, average = 25, deviation = 10):
    """
    Inputs:
        Sample parameters:
          N := total number of agents
          a := if a == True all thresholdds below 0 are converted to 0 and all thresholds above 100 are converted to 100, otherwise nothing happens
        Normal distribution parameters:
          average := average value of the normal distribution
          deviation := standard deviation of the normal distribution
    
    This function creates an array with N threshold values (0 <= x <= 100) according with a normal distribution with the given parameters
          
    Outputs:      
        A np.array with the sorted thresholds values
    
    """
    thresholds = np.array([])
    
    # generating the values
    for i in range(N):
        threshold = rd.gauss(average, deviation)     # generates a random value according with a normal distribution

        if a:
            # all thresholdds below 0 are converted to 0 and all thresholds above 100 are converted to 100
            if threshold < 0:
                threshold = 0
            elif threshold > 100:
                threshold = 100

        thresholds = np.append(thresholds, threshold)
        
    thresholds = sorted(thresholds) #sorts the array
    
    return thresholds


######################################################################################################################################################################################


def create_agents (N = 100, average = 25, deviation = 10):
    """
    Inputs:
        N := total number of agents
        average := average value of the normal distribution
        deviation := standard deviation of the normal distribution
        
        The function creates an array of Agents according with the normal distribution of threshold values
        
    Outputs:
        agents := array with all Agents
    
    """
    
    agents = np.array([])
    
    thresholds = create_thresholds(N, True, average, deviation)
    
    for i in range(N):
        agent = Agent(thresholds[i])
        agents = np.append(agents, agent)
        
    return agents


######################################################################################################################################################################################

"""
Riot simulation
"""


def simulate_riot(thresholds):
    """
    Inputs:
        thresholds := array with the Agents thresholds 
    
    This function calculates the size of an riot according with the Granovetter binary model: The Agent enters the riot if its threshold value is less or equal
    to the number (or percentage) of people rioting, otherwise, it doesn't.
    
    Outputs:
        A "answer" np.array with two elements:
            answer[0] := array with the riot's evolution over time
            answer[1] := riot's final size
    
    """
    
    riot_size = 0
    progression = np.array([0])    # array that stores the riot's evolution over time
    aux = 0
    
    while True:
        for i in range(len(thresholds)):
            if thresholds[i] <= riot_size: # check all Agents and counts which of them enters the riot
                aux += 1
        progression = np.append(progression, aux)
        if riot_size == aux:   # if the riot size doesn't change in a time step it is stable
            break
        riot_size = aux
        aux = 0
        
    return [progression, riot_size]


######################################################################################################################################################################################


def simulate_riot_stochastic(agents, steps):
    """
    Inputs:
        agents := Agents array
        steps := number of the simulation's time steps
    
    This function calculates the size of an riot according with the stochastic threshold model: The Agent has a higher probability of entering the riot if its threshold value 
    is less or equal to the number (or percentage) of people rioting, and has a low probability of entering the riot if its threshold value is less than the number (or percentage)
    of people rioting.
    
    Outputs:
        A "answer" np.array with two elements:
            answer[0] := array with the riot's evolution over time
            answer[1] := riot's final size
    
    """
    
    riot_size = 0
    progression = np.array([0])              # array that stores the riot's evolution over time
    
    for i in range(steps):
        for agent in agents:
            riot_size += agent.update_state(riot_size)
            
        progression = np.append(progression, riot_size)
        
    return [progression, riot_size]


######################################################################################################################################################################################


def simulate_riot_stochastic_2(agents, steps):
    """
    Inputs:
        agents := Agents array
        steps := number of the simulation's time steps
    
    This function calculates the size of an riot according with the stochastic threshold model: The Agent has a higher probability of entering the riot if its threshold value 
    is less or equal to the number (or percentage) of people rioting, and has a low probability of entering the riot if its threshold value is less than the number (or percentage)
    of people rioting. But if, in a time step, nothing changes the simulation stops.
    
    Outputs:
        A "answer" np.array with two elements:
            answer[0] := array with the riot's evolution over time
            answer[1] := riot's final size
    
    """
    
    riot_size = 0
    progression = np.array([0])               # array that stores the riot's evolution over time
    aux = 0
    
    for a in range(steps):
        for agent in agents:
            aux += agent.update_state(riot_size)
            
        if aux == 0:
            break
            
        riot_size += aux
        progression = np.append(progression, riot_size)
        aux = 0
        
    return [progression, riot_size]


######################################################################################################################################################################################


def simulate_riot_stochastic_exit(agents, steps):
    """
    Inputs:
        agents := Agents array
        steps := number of the simulation's time steps
    
    This function calculates the size of an riot according with the stochastic threshold model: The Agent has a higher probability of entering the riot if its threshold value is less or equal to the number (or percentage) of people rioting, and has a low probability of entering the riot if its threshold value is less than the number (or percentage) of people rioting. Moreover, there is a chance of an Agent to exit the riot according to a logistic function.
    
    Outputs:
        A "answer" np.array with two elements:
            answer[0] := array with the riot's evolution over time
            answer[1] := riot's final size
    
    """
    
    riot_size = 0
    progression = np.array([0])                                     # array that stores the riot's evolution over time
    
    for _ in range(steps):
        for agent in agents:
            riot_size += agent.update_state_exit(riot_size)

        progression = np.append(progression, riot_size)
        
    return [progression, riot_size]


######################################################################################################################################################################################


def simulate_riot_sectors(system, steps):
    """
    Inputs:
        sistema := variável da classe Sistema que contém dois arrays com N agentes cada
        passos := número de passos temporais executados pelo programa
    
    Ao receber o array com os agentes a função calcula qual o tamanho final da greve de acordo com o modelo de limiares estocastico, isto é,
    o agente possui maior probabilidade de entrar na greve caso seu limiar seja ultrapassado e, caso contrário, tem menor probabilidade de não entrar, porém
    existem dois setores nos quais as greves são calculadas independentemente. 
    
    Outpurs:
        Retorna um np.array "resposta" com duas entradas:
            resposta[0] := um array com a evolução da greve ao longo do tempo
            resposta[1] := tamanho final da greve
    
    """
    
    riots_size = np.array([0,0])
    progression = np.zeros((2,passos+1))              # array que acumula a evolução temporal da greve
    
    # verifica se os agentes quer aderir à greve 
    for i in range(1,passos+1):0
        for j in range(len(sistema.reservoir)):
            num = rnd.randint(0,1)
            if num == 0:
                riots_size[0] += sistema.sector0
        
        for j in range(sistema.sector1_size):
            tamanho_da_greve[1] += sistema.sector1[j].update_state(tamanho_da_greve[1])
            
        progressao[0][i] = tamanho_da_greve[0]
        progressao[1][i] = tamanho_da_greve[1]
            
    return [progressao, tamanho_da_greve]


######################################################################################################################################################################################


def simula_greve_setores_saida(sistema, passos):
    """
    Inputs:
        sistema := variável da classe Sistema que contém dois arrays com N agentes cada
        passos := número de passos temporais executados pelo programa
    
    Ao receber o array com os agentes a função calcula qual o tamanho final da greve de acordo com o modelo de limiares estocastico, isto é,
    o agente possui maior probabilidade de entrar na greve caso seu limiar seja ultrapassado e, caso contrário, tem menor probabilidade de não entrar, porém
    existem dois setores nos quais as greves são calculadas independentemente. Além disso, existe a possibilidade do agente sair da greve, com probabilidade
    igual à uma função logística semelhante àquela que se usa para verificar se o agente entra na greve.
    
    Outpurs:
        Retorna um np.array "resposta" com duas entradas:
            resposta[0] := um array com a evolução da greve ao longo do tempo
            resposta[1] := tamanho final da greve
    
    """
    
    tamanho_da_greve = np.array([0,0])
    progressao = np.zeros((2,passos+1))              # array que acumula a evolução temporal da greve
    
    # verifica se os agentes quer aderir à greve
    for i in range(1,passos+1):
        for j in range(sistema.sector0_size):
            tamanho_da_greve[0] += sistema.sector0[j].update_state_exit(tamanho_da_greve[0])
        
        for j in range(sistema.sector1_size):
            tamanho_da_greve[1] += sistema.sector1[j].update_state_exit(tamanho_da_greve[1])
            
        progressao[0][i] = tamanho_da_greve[0]
        progressao[1][i] = tamanho_da_greve[1]
            
    return [progressao, tamanho_da_greve]


######################################################################################################################################################################################


def simula_greve_setores_migracao(sistema, passos = 50, probabilidade_de_migracao = 0.2):
    """
    Inputs:
        sistema := variável da classe Sistema que contém dois arrays com N agentes cada
        passos := número de passos temporais executados pelo programa
        probabilidade_de_migracao := probabilidade de um agente mudar de setor do sistema
    
    Ao receber o array com os agentes a função calcula qual o tamanho final da greve de acordo com o modelo de limiares estocastico, isto é,
    o agente possui maior probabilidade de entrar na greve caso seu limiar seja ultrapassado e, caso contrário, tem menor probabilidade de não entrar, porém
    existem dois setores nos quais as greves são calculadas independentemente. Além disso, existe uma probabilidade de, aleatoriamente, um dos agentes mudar
    de setor a cada passo temporal.
    
    Outpurs:
        Retorna um np.array "resposta" com duas entradas:
            resposta[0] := um array com a evolução da greve ao longo do tempo
            resposta[1] := tamanho final da greve
            resposta[2] := número de vezes em que ocorreram migrações
    
    """
    
    tamanho_da_greve = np.array([0,0])
    progressao = np.zeros((2,passos+1))              # array que acumula a evolução temporal da greve
    migracao = 0
    
    #verifica se os agentes querem aderir à greve
    for i in range(1,passos+1):
        for j in range(sistema.sector0_size):
            tamanho_da_greve[0] += sistema.sector0[j].update_state(tamanho_da_greve[0]/sistema.sector0_size * 100)
        
        for j in range(sistema.sector1_size):       
            tamanho_da_greve[1] += sistema.sector1[j].update_state(tamanho_da_greve[1]/sistema.sector1_size * 100)
                
        progressao[0][i] = tamanho_da_greve[0]
        progressao[1][i] = tamanho_da_greve[1]
        
        #verifica se um agente quer migrar
        prob = rd.random()
        
        if prob <= probabilidade_de_migracao:
            migracao += 1
            setor = rd.randint(0,1)
            if setor == 0:
                num_agente = rd.randint(0,sistema.sector0_size-1)
                if sistema.sector0[num_agente].state == 1:
                    tamanho_da_greve[0] -= 1
                    tamanho_da_greve[1] += 1
            else:
                num_agente = rd.randint(0,sistema.sector1_size-1)
                if sistema.sector1[num_agente].state == 1:
                    tamanho_da_greve[1] -= 1
                    tamanho_da_greve[0] += 1
            
            sistema.migrate(setor, num_agente)
            
    return [progressao, tamanho_da_greve, migracao]


######################################################################################################################################################################################


def simula_greve_setores_migracao_individual(sistema, passos = 50, probabilidade_de_migracao_individual = 0.01, ligar = 0):
    """
    Inputs:
        sistema := variável da classe Sistema que contém dois arrays com N agentes cada
        passos := número de passos temporais executados pelo programa
        probabilidade_de_migracao_individual := probabilidade de um agente mudar de setor do sistema
        ligar := número do passo temporal em que as migrações passam a ser permitidas
    
    Ao receber o array com os agentes a função calcula qual o tamanho final da greve de acordo com o modelo de limiares estocastico, isto é,
    o agente possui maior probabilidade de entrar na greve caso seu limiar seja ultrapassado e, caso contrário, tem menor probabilidade de não entrar, porém
    existem dois setores nos quais as greves são calculadas independentemente. Além disso, existe uma probabilidade de, aleatoriamente, um dos agentes mudar
    de setor quando seu estado é verificado, isso para quando a migração for permitida, isto é, passo temporal atual > 'ligar'.
    
    Outpurs:
        Retorna um np.array "resposta" com duas entradas:
            resposta[0] := um array com a evolução da greve ao longo do tempo
            resposta[1] := tamanho final da greve
            resposta[2] := número de vezes em que ocorreram migrações
        
    """
    
    tamanho_da_greve = np.array([0,0])
    progressao = np.zeros((2,2*passos+1))              # array que acumula a evolução temporal da greve
    migracao = 0
    
    for i in range(1,passos+1):
        
        # verifica se os agentes querem ingressar na greve
        j = 0
        while j < sistema.sector0_size:
            tamanho_da_greve[0] += sistema.sector0[j].update_state(tamanho_da_greve[0]/sistema.sector0_size * 100)
            j += 1
        
        j = 0
        while j < sistema.sector1_size:
            tamanho_da_greve[1] += sistema.sector1[j].update_state(tamanho_da_greve[1]/sistema.sector0_size * 100)
            j += 1
        
        progressao[0][2*i - 1] = tamanho_da_greve[0]
        progressao[1][2*i - 1] = tamanho_da_greve[1]
        
        # verifica se os agentes querem migrar
        if i > ligar:
            j = 0
            while j < sistema.sector0_size:

                prob = rd.random()

                if prob <= probabilidade_de_migracao_individual:
                    migracao += 1
                    if sistema.sector0[j].state == 1:
                        tamanho_da_greve[0] -= 1
                        tamanho_da_greve[1] += 1

                    sistema.migrate(0, j)
                    j -= 1                

                j += 1

            j = 0
            while j < sistema.sector1_size:

                prob = rd.random()

                if prob <= probabilidade_de_migracao_individual:
                    migracao += 1
                    if sistema.sector1[j].state == 1:
                        tamanho_da_greve[1] -= 1
                        tamanho_da_greve[0] += 1

                    sistema.migrate(1, j)
                    j -= 1                

                j += 1
                
        progressao[0][2*i] = tamanho_da_greve[0]
        progressao[1][2*i] = tamanho_da_greve[1]
            
    return [progressao, tamanho_da_greve, migracao]


######################################################################################################################################################################################


def simula_greve_setores_migracao_individual_saida(sistema, passos = 50, probabilidade_de_migracao_individual = 0.01, ligar = 0):
    """
    Inputs:
        sistema := variável da classe Sistema que contém dois arrays com N agentes cada
        passos := número de passos temporais executados pelo programa
        probabilidade_de_migracao_individual := probabilidade de um agente mudar de setor do sistema
        ligar := número do passo temporal em que as migrações passam a ser permitidas
    
    Ao receber o array com os agentes a função calcula qual o tamanho final da greve de acordo com o modelo de limiares estocastico, isto é,
    o agente possui maior probabilidade de entrar na greve caso seu limiar seja ultrapassado e, caso contrário, tem menor probabilidade de não entrar, porém
    existem dois setores nos quais as greves são calculadas independentemente. Além disso, existe uma probabilidade de, aleatoriamente, um dos agentes mudar
    de setor quando seu estado é verificado, isso para quando a migração for permitida, isto é, passo temporal atual > 'ligar'. Ainda, existe a possibilidade
    do agente sair da greve, com probabilidade igual à uma função logística semelhante àquela que se usa para verificar se o agente entra na greve.
    
    Outpurs:
        Retorna um np.array "resposta" com duas entradas:
            resposta[0] := um array com a evolução da greve ao longo do tempo
            resposta[1] := tamanho final da greve
            resposta[2] := número de vezes em que ocorreram migrações
        
    """
    
    tamanho_da_greve = np.array([0,0])
    progressao = np.zeros((2,2*passos+1))              # array que acumula a evolução temporal da greve
    migracao = 0
    
    for i in range(1,passos+1):
        
        # verifica se os agentes querem ingressar na greve
        j = 0
        while j < sistema.sector0_size:
            tamanho_da_greve[0] += sistema.sector0[j].update_state_exit(tamanho_da_greve[0]/sistema.sector0_size * 100)
            j += 1
        
        j = 0
        while j < sistema.sector1_size:
            tamanho_da_greve[1] += sistema.sector1[j].update_state_exit(tamanho_da_greve[1]/sistema.sector0_size * 100)
            j += 1
        
        progressao[0][2*i - 1] = tamanho_da_greve[0]
        progressao[1][2*i - 1] = tamanho_da_greve[1]
        
        # verifica se os agentes querem migrar
        if i > ligar:
            j = 0
            while j < sistema.sector0_size:

                prob = rd.random()

                if prob <= probabilidade_de_migracao_individual:
                    migracao += 1
                    if sistema.sector0[j].state == 1:
                        tamanho_da_greve[0] -= 1
                        tamanho_da_greve[1] += 1

                    sistema.migrate(0, j)
                    j -= 1                

                j += 1

            j = 0
            while j < sistema.sector1_size:

                prob = rd.random()

                if prob <= probabilidade_de_migracao_individual:
                    migracao += 1
                    if sistema.sector1[j].state == 1:
                        tamanho_da_greve[1] -= 1
                        tamanho_da_greve[0] += 1

                    sistema.migrate(1, j)
                    j -= 1                

                j += 1
                
        progressao[0][2*i] = tamanho_da_greve[0]
        progressao[1][2*i] = tamanho_da_greve[1]
            
    return [progressao, tamanho_da_greve, migracao]


######################################################################################################################################################################################

def funcao_migracao_gregario(sectorA_size, sectorB_size):
    """
    Inputs:
        sectorA_size := tamanho da greve no setor em que o agente se encontra
        sectorB_size := tamanho da greve no outro setor
        
    Outputs:
        Retorna um float, sendo este a probabilidade de o agente mudar de setor
        
    """
    
    dif = sectorB_size - sectorA_size
    
    if dif <= 0:
        return 0
    
    else:
        m = 3*10e-5
        res = 1 - np.exp(-m * dif)
        return res


######################################################################################################################################################################################


def simula_greve_setores_migracao_individual_gregario(sistema, passos = 50, ligar = 0):
    """
    Inputs:
        sistema := variável da classe Sistema que contém dois arrays com N agentes cada
        passos := número de passos temporais executados pelo programa 
        ligar := número do passo temporal em que as migrações passam a ser permitidas
    
    Ao receber o array com os agentes a função calcula qual o tamanho final da greve de acordo com o modelo de limiares estocastico, isto é,
    o agente possui maior probabilidade de entrar na greve caso seu limiar seja ultrapassado e, caso contrário, tem menor probabilidade de não entrar, porém
    existem dois setores nos quais as greves são calculadas independentemente. Além disso, existe uma probabilidade de, aleatoriamente, um dos agentes mudar
    de setor quando seu estado é verificado, isso para quando a migração for permitida, isto é, passo temporal atual > 'ligar'. Os agentes apenas migram
    para o setor com maior número de agentes na greve, de acordo com a diferença de agentes em greve em cada setor.
    
    Outpurs:
        Retorna um np.array "resposta" com duas entradas:
            resposta[0] := um array com a evolução da greve ao longo do tempo
            resposta[1] := tamanho final da greve
            resposta[2] := número de vezes em que ocorreram migrações
        
    """
    
    tamanho_da_greve = np.array([0,0])
    progressao = np.zeros((2,2*passos+1))              # array que acumula a evolução temporal da greve
    migracao = 0
    
    for i in range(1,passos+1):
        
        # verifica se os agentes querem ingressar na greve
        j = 0
        while j < sistema.sector0_size:
            tamanho_da_greve[0] += sistema.sector0[j].update_state(tamanho_da_greve[0]/sistema.sector0_size * 100)
            j += 1
        
        j = 0
        while j < sistema.sector1_size:
            tamanho_da_greve[1] += sistema.sector1[j].update_state(tamanho_da_greve[1]/sistema.sector0_size * 100)
            j += 1
        
        progressao[0][2*i - 1] = tamanho_da_greve[0]
        progressao[1][2*i - 1] = tamanho_da_greve[1]
        
        # verifica se os agentes querem migrar
        if i > ligar:
            j = 0
            while j < sistema.sector0_size:

                prob = rd.random()

                if prob < funcao_migracao_gregario(tamanho_da_greve[0],tamanho_da_greve[1]):
                    migracao += 1
                    if sistema.sector0[j].state == 1:
                        tamanho_da_greve[0] -= 1
                        tamanho_da_greve[1] += 1

                    sistema.migrate(0, j)
                    j -= 1                

                j += 1

            j = 0
            while j < sistema.sector1_size:

                prob = rd.random()

                if prob < funcao_migracao_gregario(tamanho_da_greve[1],tamanho_da_greve[0]):
                    migracao += 1
                    if sistema.sector1[j].state == 1:
                        tamanho_da_greve[1] -= 1
                        tamanho_da_greve[0] += 1

                    sistema.migrate(1, j)
                    j -= 1                

                j += 1
                
        progressao[0][2*i] = tamanho_da_greve[0]
        progressao[1][2*i] = tamanho_da_greve[1]
            
    return [progressao, tamanho_da_greve, migracao]


######################################################################################################################################################################################


def simula_greve_setores_migracao_individual_gregario_saida(sistema, passos = 50, ligar = 0):
    """
    Inputs:
        sistema := variável da classe Sistema que contém dois arrays com N agentes cada
        passos := número de passos temporais executados pelo programa 
        ligar := número do passo temporal em que as migrações passam a ser permitidas
    
    Ao receber o array com os agentes a função calcula qual o tamanho final da greve de acordo com o modelo de limiares estocastico, isto é,
    o agente possui maior probabilidade de entrar na greve caso seu limiar seja ultrapassado e, caso contrário, tem menor probabilidade de não entrar, porém
    existem dois setores nos quais as greves são calculadas independentemente. Além disso, existe uma probabilidade de, aleatoriamente, um dos agentes mudar
    de setor quando seu estado é verificado, isso para quando a migração for permitida, isto é, passo temporal atual > 'ligar'. Os agentes apenas migram
    para o setor com maior número de agentes na greve, de acordo com a diferença de agentes em greve em cada setor. Ainda, existe a possibilidade
    do agente sair da greve, com probabilidade igual à uma função logística semelhante àquela que se usa para verificar se o agente entra na greve.
    
    Outpurs:
        Retorna um np.array "resposta" com duas entradas:
            resposta[0] := um array com a evolução da greve ao longo do tempo
            resposta[1] := tamanho final da greve
            resposta[2] := número de vezes em que ocorreram migrações
        
    """
    
    tamanho_da_greve = np.array([0,0])
    progressao = np.zeros((2,2*passos+1))              # array que acumula a evolução temporal da greve
    migracao = 0
    
    for i in range(1,passos+1):
        
        # verifica se os agentes querem ingressar na greve
        j = 0
        while j < sistema.sector0_size:
            tamanho_da_greve[0] += sistema.sector0[j].update_state_exit(tamanho_da_greve[0]/sistema.sector0_size * 100)
            j += 1
        
        j = 0
        while j < sistema.sector1_size:
            tamanho_da_greve[1] += sistema.sector1[j].update_state_exit(tamanho_da_greve[1]/sistema.sector0_size * 100)
            j += 1
        
        progressao[0][2*i - 1] = tamanho_da_greve[0]
        progressao[1][2*i - 1] = tamanho_da_greve[1]
        
        # verifica se os agentes querem migrar
        if i > ligar:
            j = 0
            while j < sistema.sector0_size:

                prob = rd.random()

                if prob < funcao_migracao_gregario(tamanho_da_greve[0],tamanho_da_greve[1]):
                    migracao += 1
                    if sistema.sector0[j].state == 1:
                        tamanho_da_greve[0] -= 1
                        tamanho_da_greve[1] += 1

                    sistema.migrate(0, j)
                    j -= 1                

                j += 1

            j = 0
            while j < sistema.sector1_size:

                prob = rd.random()

                if prob < funcao_migracao_gregario(tamanho_da_greve[1],tamanho_da_greve[0]):
                    migracao += 1
                    if sistema.sector1[j].state == 1:
                        tamanho_da_greve[1] -= 1
                        tamanho_da_greve[0] += 1

                    sistema.migrate(1, j)
                    j -= 1                

                j += 1
                
        progressao[0][2*i] = tamanho_da_greve[0]
        progressao[1][2*i] = tamanho_da_greve[1]
            
    return [progressao, tamanho_da_greve, migracao]