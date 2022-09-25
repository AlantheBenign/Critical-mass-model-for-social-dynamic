import numpy as np
import matplotlib.pyplot as plt
import random as rd

######################################################################################################################################################################################

"""
This class is used to create the simulation enviroment. 

"""

class System:
    
    def __init__(self, agents, limit_0, limit_1):
        self.reservoir = agents
        self.reservoir_size = len(agents)
        self.place0 = np.array([])
        self.place0_size = limet_0
        self.place1 = np.array([])
        self.place1_size = limit_1
        
    def migrate(self, place, i):
        if place == 0:
            agent = self.place0[i]
            self.place0 = np.delete(self.place0, i)
            self.place0_size -= 1
            self.place1 = np.append(self.place1, agent)
            self.place1_size += 1
            
        else:
            agent = self.place1[i]
            self.place1 = np.delete(self.place1, i)
            self.place1_size -= 1
            self.place0 = np.append(self.place0, agent)
            self.place0_size += 1
            
            
######################################################################################################################################################################################

"""
This class is used to store an agent's data. 

Every agent has a threshold and a state value, if the state value is 1 the agent is part of the riot, for exemple.
We are using the stochastic method.

agent.place == -1: it is in the reservoir; agent.place == 0: it is in the place_0; agent.place == 1: it is in the place_1. 
"""


class Agent:
    
    def __init__(self, threshold):
        self.threshold = threshold
        self.state = 0
        self.place = -1
    
    
    def threshold_model(self, percentage):
        m = 0.2                                                                # if m -> inf, the model approaches the Granovetter's binary model of thresholds.
        probability = 1 / (1 + np.exp( m * (self.threshold - percentage) ) )   # stochastic model of thresholds
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
        agent = Agente(thresholds[i])
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
    
    This function calculates the size of an riot according with the stochastic threshold model: The Agent has a higher probability of entering the riot if its threshold value 
    is less or equal to the number (or percentage) of people rioting, and has a low probability of entering the riot if its threshold value is less than the number (or percentage)
    of people rioting. Moreover, there is a chance of an Agent to exit the riot according to a logistic function.
    
    Outputs:
        A "answer" np.array with two elements:
            answer[0] := array with the riot's evolution over time
            answer[1] := riot's final size
    
    """
    
    riot_size = 0
    progression = np.array([0])                                     # array that stores the riot's evolution over time
    
    for _ in range(steps):
        for agent in agents:
            riot_size += agente.update_state_exit(riot_size)

        progression = np.append(progression, riot_size)
        
    return [progression, riot_size]


######################################################################################################################################################################################


def change_place(agent):
    """
    Inputs:
        agent := variable of the Agent class
    
    This function changes the Agent's place.
    
    """
    
    if agente.place == 0:
        agente.place = 1
    else:
        agente.place = 0
        
           
######################################################################################################################################################################################


def simula_greve_setores(sistema, passos):
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
    
    tamanho_da_greve = np.array([0,0])
    progressao = np.zeros((2,passos+1))              # array que acumula a evolução temporal da greve
    
    # verifica se os agentes quer aderir à greve
    for i in range(1,passos+1):
        for j in range(sistema.place0_size):
            tamanho_da_greve[0] += sistema.place0[j].update_state(tamanho_da_greve[0])
        
        for j in range(sistema.place1_size):
            tamanho_da_greve[1] += sistema.place1[j].update_state(tamanho_da_greve[1])
            
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
        for j in range(sistema.place0_size):
            tamanho_da_greve[0] += sistema.place0[j].update_state_exit(tamanho_da_greve[0])
        
        for j in range(sistema.place1_size):
            tamanho_da_greve[1] += sistema.place1[j].update_state_exit(tamanho_da_greve[1])
            
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
        for j in range(sistema.place0_size):
            tamanho_da_greve[0] += sistema.place0[j].update_state(tamanho_da_greve[0]/sistema.place0_size * 100)
        
        for j in range(sistema.place1_size):       
            tamanho_da_greve[1] += sistema.place1[j].update_state(tamanho_da_greve[1]/sistema.place1_size * 100)
                
        progressao[0][i] = tamanho_da_greve[0]
        progressao[1][i] = tamanho_da_greve[1]
        
        #verifica se um agente quer migrar
        prob = rd.random()
        
        if prob <= probabilidade_de_migracao:
            migracao += 1
            setor = rd.randint(0,1)
            if setor == 0:
                num_agente = rd.randint(0,sistema.place0_size-1)
                if sistema.place0[num_agente].state == 1:
                    tamanho_da_greve[0] -= 1
                    tamanho_da_greve[1] += 1
            else:
                num_agente = rd.randint(0,sistema.place1_size-1)
                if sistema.place1[num_agente].state == 1:
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
        while j < sistema.place0_size:
            tamanho_da_greve[0] += sistema.place0[j].update_state(tamanho_da_greve[0]/sistema.place0_size * 100)
            j += 1
        
        j = 0
        while j < sistema.place1_size:
            tamanho_da_greve[1] += sistema.place1[j].update_state(tamanho_da_greve[1]/sistema.place0_size * 100)
            j += 1
        
        progressao[0][2*i - 1] = tamanho_da_greve[0]
        progressao[1][2*i - 1] = tamanho_da_greve[1]
        
        # verifica se os agentes querem migrar
        if i > ligar:
            j = 0
            while j < sistema.place0_size:

                prob = rd.random()

                if prob <= probabilidade_de_migracao_individual:
                    migracao += 1
                    if sistema.place0[j].state == 1:
                        tamanho_da_greve[0] -= 1
                        tamanho_da_greve[1] += 1

                    sistema.migrate(0, j)
                    j -= 1                

                j += 1

            j = 0
            while j < sistema.place1_size:

                prob = rd.random()

                if prob <= probabilidade_de_migracao_individual:
                    migracao += 1
                    if sistema.place1[j].state == 1:
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
        while j < sistema.place0_size:
            tamanho_da_greve[0] += sistema.place0[j].update_state_exit(tamanho_da_greve[0]/sistema.place0_size * 100)
            j += 1
        
        j = 0
        while j < sistema.place1_size:
            tamanho_da_greve[1] += sistema.place1[j].update_state_exit(tamanho_da_greve[1]/sistema.place0_size * 100)
            j += 1
        
        progressao[0][2*i - 1] = tamanho_da_greve[0]
        progressao[1][2*i - 1] = tamanho_da_greve[1]
        
        # verifica se os agentes querem migrar
        if i > ligar:
            j = 0
            while j < sistema.place0_size:

                prob = rd.random()

                if prob <= probabilidade_de_migracao_individual:
                    migracao += 1
                    if sistema.place0[j].state == 1:
                        tamanho_da_greve[0] -= 1
                        tamanho_da_greve[1] += 1

                    sistema.migrate(0, j)
                    j -= 1                

                j += 1

            j = 0
            while j < sistema.place1_size:

                prob = rd.random()

                if prob <= probabilidade_de_migracao_individual:
                    migracao += 1
                    if sistema.place1[j].state == 1:
                        tamanho_da_greve[1] -= 1
                        tamanho_da_greve[0] += 1

                    sistema.migrate(1, j)
                    j -= 1                

                j += 1
                
        progressao[0][2*i] = tamanho_da_greve[0]
        progressao[1][2*i] = tamanho_da_greve[1]
            
    return [progressao, tamanho_da_greve, migracao]


######################################################################################################################################################################################

def funcao_migracao_gregario(placeA_size, placeB_size):
    """
    Inputs:
        placeA_size := tamanho da greve no setor em que o agente se encontra
        placeB_size := tamanho da greve no outro setor
        
    Outputs:
        Retorna um float, sendo este a probabilidade de o agente mudar de setor
        
    """
    
    dif = placeB_size - placeA_size
    
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
        while j < sistema.place0_size:
            tamanho_da_greve[0] += sistema.place0[j].update_state(tamanho_da_greve[0]/sistema.place0_size * 100)
            j += 1
        
        j = 0
        while j < sistema.place1_size:
            tamanho_da_greve[1] += sistema.place1[j].update_state(tamanho_da_greve[1]/sistema.place0_size * 100)
            j += 1
        
        progressao[0][2*i - 1] = tamanho_da_greve[0]
        progressao[1][2*i - 1] = tamanho_da_greve[1]
        
        # verifica se os agentes querem migrar
        if i > ligar:
            j = 0
            while j < sistema.place0_size:

                prob = rd.random()

                if prob < funcao_migracao_gregario(tamanho_da_greve[0],tamanho_da_greve[1]):
                    migracao += 1
                    if sistema.place0[j].state == 1:
                        tamanho_da_greve[0] -= 1
                        tamanho_da_greve[1] += 1

                    sistema.migrate(0, j)
                    j -= 1                

                j += 1

            j = 0
            while j < sistema.place1_size:

                prob = rd.random()

                if prob < funcao_migracao_gregario(tamanho_da_greve[1],tamanho_da_greve[0]):
                    migracao += 1
                    if sistema.place1[j].state == 1:
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
        while j < sistema.place0_size:
            tamanho_da_greve[0] += sistema.place0[j].update_state_exit(tamanho_da_greve[0]/sistema.place0_size * 100)
            j += 1
        
        j = 0
        while j < sistema.place1_size:
            tamanho_da_greve[1] += sistema.place1[j].update_state_exit(tamanho_da_greve[1]/sistema.place0_size * 100)
            j += 1
        
        progressao[0][2*i - 1] = tamanho_da_greve[0]
        progressao[1][2*i - 1] = tamanho_da_greve[1]
        
        # verifica se os agentes querem migrar
        if i > ligar:
            j = 0
            while j < sistema.place0_size:

                prob = rd.random()

                if prob < funcao_migracao_gregario(tamanho_da_greve[0],tamanho_da_greve[1]):
                    migracao += 1
                    if sistema.place0[j].state == 1:
                        tamanho_da_greve[0] -= 1
                        tamanho_da_greve[1] += 1

                    sistema.migrate(0, j)
                    j -= 1                

                j += 1

            j = 0
            while j < sistema.place1_size:

                prob = rd.random()

                if prob < funcao_migracao_gregario(tamanho_da_greve[1],tamanho_da_greve[0]):
                    migracao += 1
                    if sistema.place1[j].state == 1:
                        tamanho_da_greve[1] -= 1
                        tamanho_da_greve[0] += 1

                    sistema.migrate(1, j)
                    j -= 1                

                j += 1
                
        progressao[0][2*i] = tamanho_da_greve[0]
        progressao[1][2*i] = tamanho_da_greve[1]
            
    return [progressao, tamanho_da_greve, migracao]