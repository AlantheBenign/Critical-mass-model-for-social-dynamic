import numpy as np
import matplotlib.pyplot as plt
import random as rd

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
    
    def threshold_model(self, percentage):
        m = 0.2                                                                # caso m -> inf, então o modelo se aproxima do modelo binário de limiares de Granovetter
        probability = 1 / (1 + np.exp( m * (self.threshold - percentage) ) )   # modelo estocástico de limiares
        return probability
        
    def update_state(self, percentage):
        rnd = rd.random()
        if rnd <= self.threshold_model(percentage):
            self.state = 1
            
    def update_state_exit(self, percentage):
        rnd = rd.random()
        if rnd <= self.threshold_model(percentage):
            self.state = 1
        else:
            self.state = 0

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
    
    Ao receber o array com os agente a função calcula qual o tamanho final da greve de acordo com o modelo de limiares estocastico, isto é,
    o agente possui maior probabilidade de entrar na greve caso seu limiar seja ultrapassado e, caso contrário, tem menor probabilidade de não entrar.
    
    Outputs:
        Retorna um np.array "resposta" com duas entradas:
            resposta[0] := um array com a evolução da greve ao longo do tempo
            resposta[1] := tamanho final da greve
    
    """
    
    tamanho_da_greve = 0
    progressao = np.array([0])                                    # array que acumula a evolução temporal da greve
    aux = 0
    
    for a in range(passos):
        for agente in agentes:
            agente.update_state(tamanho_da_greve)
            if agente.state == 1:
                aux += 1
        tamanho_da_greve = aux
        progressao = np.append(progressao, tamanho_da_greve)
        aux = 0
        
    return [progressao, tamanho_da_greve]


######################################################################################################################################################################################


def simula_greve_estocastico_2(agentes, passos):
    """
    Inputs:
        N := número de agentes
        media := média da distribuição normal
        desvio_padrao := desvio padrão da distriubição normal
    
    Ao receber os parâmetros da distruibuição a função gera os agentes e calcula qual o tamanho final da greve de acordo com o modelo de limiares estocastico, isto é,
    o agente possui maior probabilidade de entrar na greve caso seu limiar seja ultrapassado e, caso contrário, tem menor probabilidade de não entrar.
    Entretanto, caso do passo x para o passo x+1 não houver alteração no sistema, o algorítmo para.
    
    Outputs:
        Retorna um np.array "resposta" com duas entradas:
            resposta[0] := um array com a evolução da greve ao longo do tempo
            resposta[1] := tamanho final da greve
    
    """
    
    tamanho_da_greve = 0
    progressao = np.array([0])                                    # array que acumula a evolução temporal da greve
    aux = 0
    
    for a in range(passos):
        for agente in agentes:
            agente.update_state(tamanho_da_greve)
            if agente.state == 1:
                aux += 1
        if aux == 0:
            break
        tamanho_da_greve = aux
        progressao = np.append(progressao, tamanho_da_greve)
        aux = 0
        
    return [progressao, tamanho_da_greve]

######################################################################################################################################################################################


def simula_greve_estocastico_saida(agentes, passos):
    """
    Inputs:
        N := número de agentes
        media := média da distribuição normal
        desvio_padrao := desvio padrão da distriubição normal
    
    Ao receber os parâmetros da distruibuição a função gera os agentes e calcula qual o tamanho final da greve de acordo com o modelo de limiares estocastico, isto é,
    o agente possui maior probabilidade de entrar na greve caso seu limiar seja ultrapassado e, caso contrário, tem menor probabilidade de não entrar.
    Entretanto, caso do passo x para o passo x+1 não houver alteração no sistema, o algorítmo para.
    
    Outputs:
        Retorna um np.array "resposta" com duas entradas:
            resposta[0] := um array com a evolução da greve ao longo do tempo
            resposta[1] := tamanho final da greve
    
    """
    
    tamanho_da_greve = 0
    progressao = np.array([0])                                    # array que acumula a evolução temporal da greve
    aux = 0
    change = 0
    
    for a in range(passos):
        for agente in agentes:
            agente.update_state_exit(tamanho_da_greve)
            if agente.state == 1:
                aux += 1
                change = 1
            else:
                aux -= 1
                change = 1
        if aux == 0 and change == 0:
            break
        tamanho_da_greve = aux
        progressao = np.append(progressao, tamanho_da_greve)
        aux = 0
        change = 0
        
    return [progressao, tamanho_da_greve]

######################################################################################################################################################################################


#def 