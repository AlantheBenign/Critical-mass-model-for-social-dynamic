import numpy as np
import matplotlib.pyplot as plt
import random as rd

#trocar arrays para sets

######################################################################################################################################################################################

"""
Criando o sistema de setores para simular a greve. 

"""

class Sistema:
    
    def __init__(self, agentes0, agentes1):
        self.place0_size = len(agentes0)
        self.place0 = agentes0
        self.place1_size = len(agentes1)
        self.place1 = agentes1
        
    def migrate(self, placeA, i):
        if placeA == 0:
            agente = self.place0[i]
            self.place0 = np.delete(self.place0, i)
            self.place0_size -= 1
            self.place1 = np.append(self.place1, agente)
            self.place1_size += 1
            
        else:
            agente = self.place1[i]
            self.place1 = np.delete(self.place1, i)
            self.place1_size -= 1
            self.place0 = np.append(self.place0, agente)
            self.place0_size += 1
            
            
######################################################################################################################################################################################

"""
Criando os agentes para simular o sistema. 

Todo agente possui seu limiar e também seu estado, se está na greve ou não, por exemplo.
Aqui utilizamos um modelo estocástico para os limiares.
"""


class Agente:
    
    
    def __init__(self, threshold):
        self.threshold = threshold
        self.state = 0
        self.place = 0
    
    
    def threshold_model(self, percentage):
        m = 0.2                                                                # caso m -> inf, então o modelo se aproxima do modelo binário de limiares de Granovetter
        probability = 1 / (1 + np.exp( m * (self.threshold - percentage) ) )   # modelo estocástico de limiares
        return probability
        
        
    # atualiza o estado de um agente segundo seu limiar e retorna 1 se o agente aderir à greve e 0 se nada mudar em seu estado original         
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

        
    # atualiza o estado de um agente segundo seu limiar e retorna 1 se o agente aderir à greve e 0 se nada mudar em seu estado original e -1 se ele sair da greve        
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
Gerando os limiares de um espaço amostral.

A população de origem da amostra possui uma distribuição normal de limiares entre seus agentes.
"""


def cria_thresholds(N = 100, a = True, media = 25, desvio_padrao = 10):
    """
    Inputs:
        Parâmetros do espaço amostral:
          N := número total de agentes
          a := se a == True os limiares menores que 0 se tornam 0 e os maiores que 100 se tornam 100, caso contrário nada acontece
        Parâmetros da distribuição gaussiana:
          media := média da distribuição gaussiana
          desvio_padrao := desvio padrão da distribuição gaussiana
    
    Essa função cria um array com N valores de threshold (threshold entre 0 e 100) segundo uma distribuição normal de acordo com os parâmetros fornecidos.
          
    Outputs:      
        Retorna um np.array com os thresholds de todos os agentes de forma ordenada
    
    """
    thresholds = np.array([])
    
    # gerando os limiares
    for i in range(N):
        limiar = rd.gauss(media, desvio_padrao)     # gera número aleatório numa distribuição normal

        if a:
            # todos os thresholds negativos se tornam 0 e todos aqueles maiores que 100 se tornam 100 (por coerência com a teoria)
            if limiar < 0:
                limiar = 0
            elif limiar > 100:
                limiar = 100

        thresholds = np.append(thresholds, limiar)
        
    thresholds = sorted(thresholds) #ordena o array de thresholds
    
    return thresholds


######################################################################################################################################################################################


def cria_agentes(N = 100, media = 25, desvio_padrao = 10):
    """
    Inputs:
        N := número de agentes que se vai criar
        media := média da distribuição gaussiana
        desvio_padrao := desvio padrão da distribuição gaussiana
        
        A função criar um array de agentes criados a partir da geração de limiares segundo uma distribuição gaussiana.
        
    Outputs:
        agentes := array com todos os agentes criados
    
    """
    
    agentes = np.array([])
    
    thresholds = cria_thresholds(N, True, media, desvio_padrao)
    
    for i in range(N):
        agente = Agente(thresholds[i])
        agentes = np.append(agentes, agente)
        
    return agentes


######################################################################################################################################################################################

"""
Cálculo do tamanho final da greve
"""


def simula_greve(thresholds):
    """
    Inputs:
        thresholds := array com os thresholds de todos os agentes
    
    Ao receber o array com os thresholds de todos os agentes a função calcula qual o tamanho final da greve de acordo com o modelo de limiares binário, isto é,
    o agente com certeza entra na greve caso seu limiar seja ultrapassado e, caso contrário, não entra.
    
    Outputs:
        Retorna um np.array "resposta" com duas entradas:
            resposta[0] := um array com a evolução da greve ao longo do tempo
            resposta[1] := tamanho final da greve
    
    """
    
    tamanho_da_greve = 0
    progressao = np.array([0])    # array que acumula a evolução temporal da greve
    aux = 0
    
    while True:
        for i in range(len(thresholds)):
            if thresholds[i] <= tamanho_da_greve: # verifica os thresholds de todas os agentes e conta quantos entram para a greve
                aux += 1
        progressao = np.append(progressao, aux)
        if tamanho_da_greve == aux:   # caso o tamanho da greve não mude de um passo temporal para outro ela está estável
            break
        tamanho_da_greve = aux
        aux = 0
        
    return [progressao, tamanho_da_greve]


######################################################################################################################################################################################


def simula_greve_estocastico(agentes, passos):
    """
    Inputs:
        agentes := array com todos os agentes
        passos := número de passos temporais executados pelo algorítmo
    
    Ao receber o array com os agentes a função calcula qual o tamanho final da greve de acordo com o modelo de limiares estocastico, isto é,
    o agente possui maior probabilidade de entrar na greve caso seu limiar seja ultrapassado e, caso contrário, tem menor probabilidade de não entrar.
    
    Outputs:
        Retorna um np.array "resposta" com duas entradas:
            resposta[0] := um array com a evolução da greve ao longo do tempo
            resposta[1] := tamanho final da greve
    
    """
    
    tamanho_da_greve = 0
    progressao = np.array([0])              # array que acumula a evolução temporal da greve
    
    for i in range(passos):
        for agente in agentes:
            tamanho_da_greve += agente.update_state(tamanho_da_greve)
            
        progressao = np.append(progressao, tamanho_da_greve)
        
    return [progressao, tamanho_da_greve]


######################################################################################################################################################################################


def simula_greve_estocastico_2(agentes, passos):
    """
    Inputs:
        agentes := array com todos os agentes
        passos := número de passos temporais executados pelo algorítmo
    
    Ao receber o array com os agentes a função calcula qual o tamanho final da greve de acordo com o modelo de limiares estocastico, isto é,
    o agente possui maior probabilidade de entrar na greve caso seu limiar seja ultrapassado e, caso contrário, tem menor probabilidade de não entrar.
    Entretanto, caso do passo x para o passo x+1 não houver alteração no sistema, o algorítmo para.
    
    Outputs:
        Retorna um np.array "resposta" com duas entradas:
            resposta[0] := um array com a evolução da greve ao longo do tempo
            resposta[1] := tamanho final da greve
    
    """
    
    tamanho_da_greve = 0
    progressao = np.array([0])              # array que acumula a evolução temporal da greve
    aux = 0
    
    for a in range(passos):
        for agente in agentes:
            aux += agente.update_state(tamanho_da_greve)
            
        if aux == 0:
            break
            
        tamanho_da_greve += aux
        progressao = np.append(progressao, tamanho_da_greve)
        aux = 0
        
    return [progressao, tamanho_da_greve]


######################################################################################################################################################################################


def simula_greve_estocastico_saida(agentes, passos):
    """
    Inputs:
        agentes := array com todos os agentes
        passos := número de passos temporais executados pelo algorítmo
    
    Ao receber o array com os agentes a função calcula qual o tamanho final da greve de acordo com o modelo de limiares estocastico, isto é,
    o agente possui maior probabilidade de entrar na greve caso seu limiar seja ultrapassado e, caso contrário, tem menor probabilidade de não entrar, além disso,
    existe a possibilidade do agente sair da greve, com probabilidade igual à uma função logística semelhante àquela que se usa para verificar se o agente entra na greve.
    
    Outputs:
        Retorna um np.array "resposta" com duas entradas:
            resposta[0] := um array com a evolução da greve ao longo do tempo
            resposta[1] := tamanho final da greve
    
    """
    
    tamanho_da_greve = 0
    progressao = np.array([0])                                    # array que acumula a evolução temporal da greve
    
    for _ in range(passos):
        for agente in agentes:
            tamanho_da_greve += agente.update_state_exit(tamanho_da_greve)

        progressao = np.append(progressao, tamanho_da_greve)
        
    return [progressao, tamanho_da_greve]


######################################################################################################################################################################################


def change_place(agente):
    """
    Inputs:
        agente := variável da classe Agente
    
    A função troca o agente de setor da greve.
    
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