import inspect
import os
import sys
import gym
import unittest
import GPUtil
from time import time
from multiprocessing import Process
import psutil

from urnai.utils.logger import Logger
from urnai.agents.generic_agent import GenericAgent
from urnai.agents.rewards.gym import FrozenlakeReward
from urnai.agents.states.gym import FrozenLakeState
from urnai.envs.gym import GymEnv
from urnai.models.ddqn_keras import DDQNKeras
from urnai.models.model_builder import ModelBuilder
from urnai.trainers.trainer import Trainer


# class GymAgent:
#     def __init__(self, env_name):
#         self.env = gym.make(env_name)
#         self.observation_space = self.env.observation_space
#         self.action_space = self.env.action_space
#         self.observation = None
#         self.action = None
#         self.reward = None
#         self.done = False
#         self.ep_count = 0
        

#     def reset(self):
#         self.observation = self.env.reset()
#         self.action = None
#         self.reward = None
#         self.done = False
#         return self.observation

#     def act(self, action):
#         self.action = action
#         self.observation, self.reward, self.done, _ = self.env.step(self.action)
#         return self.reward

#     def observe(self, reward=None):
#         return self.observation, self.reward, self.done
    
#     def record_episode(self, ep_reward, has_won, steps_count, agent_info, ep_actions):
#         self.ep_count += 1


#         #registra as ações tomadas pelo agente e calcula a média das ações tomadas até aquele ponto.
#         for i in range(self.agent_action_size):

#             #adiciona a ação do episódio atual
#             self.ep_agent_actions[i].append(ep_actions[i]) 
#             #calcula e adiciona a média das ações tomadas até este ponto.
#             self.avg_ep_agent_actions[i].append(sum(self.ep_agent_actions[i]) / self.ep_count)


#         #armazena a recompensa obtida no episódio atual
#         self.ep_rewards.append(ep_reward)
#         #calcula e armazena a média das recompensas obtidas até este episódio
#         self.ep_avg_rewards.append(sum(self.ep_rewards) / self.ep_count)

#         #adciona a contagem e a média da contagem de etapas que indica a quantidade de ações que o agente tomou durante um episódio.
#         self.ep_steps_count.append(steps_count)
#         self.ep_avg_steps.append(sum(self.ep_steps_count) / self.ep_count)

#         # time and sps stuff
#         #Essas métricas podem ajudar a avaliar o desempenho do agente ao longo do tempo e a identificar possíveis 
#         # problemas ou melhorias que precisam ser feitas no modelo
#         #Essa parte do método calcula e armazena as métricas relacionadas 
#         # à duração e ao desempenho do agente durante o episódio.

#         #mede quanto tempo o episódio levou para ser concluído (diferença entre o tempo atual e o tempo de início do ep).
#         episode_duration = time() - self.episode_temp_start_time 
#         #armazena a duração do episódio na lista que armazena a duração de cada episódio
#         self.episode_duration_list.append(round(episode_duration, 1))

#         if episode_duration != 0:
#             #número de etapas do agente por segundo em cada episódio
#             self.episode_sps_list.append(round(steps_count / episode_duration, 2))
#         else:
#             self.episode_sps_list.append(0)

#         #armazena a média do número de etapas por segundo para todos os episódios concluídos até agora.        
#         self.avg_sps_list.append(round(sum(self.episode_sps_list) / self.ep_count, 2))



#         # performance stuff
#         #uso da biblioteca psutil para coletar 
#         # informações de uso do sistema em relação à CPU e à memória.

#         #coleta a porcentagem de uso de memória atual do sistema.
#         memory_usage_percent = psutil.virtual_memory().percent

#         #calcula a porcentagem de memória disponível em relação ao total.
#         memory_avail_percent = (psutil.virtual_memory().available * 100 /
#                                 psutil.virtual_memory().total)
        
#         #calcula a quantidade de memória em uso em gigabytes.
#         memory_usage_gigs = psutil.virtual_memory().used / 1024 ** 3
#         #calcula a quantidade de memória disponível em gigabytes.
#         memory_avail_gigs = psutil.virtual_memory().free / 1024 ** 3
#         #coleta a porcentagem de uso da CPU no momento.
#         cpu_usage_percent = psutil.cpu_percent()

#         # has GPU available
#         #Essa parte do código é responsável por tentar obter informações sobre a utilização da GPU (unidade de processamento gráfico) no sistema. 
#         # Ele usa a biblioteca GPUtil para tentar acessar informações sobre a memória e a carga da GPU. 
#         # Se conseguir acessar essas informações, ele armazena a porcentagem de uso de memória da GPU, o uso de memória da GPU em gigabytes e a porcentagem de uso da GPU. 
#         # Caso contrário, ele define todos esses valores como 0.
#         try:
#             gpu_memory_usage_percent = GPUtil.getGPUs()[0].memoryUtil * 100
#             gpu_memory_usage_gigs = GPUtil.getGPUs()[0].memoryUsed / 1000
#             gpu_usage_percent = GPUtil.getGPUs()[0].load * 100

#         except Exception:
#             gpu_memory_usage_percent = 0
#             gpu_memory_usage_gigs = 0
#             gpu_usage_percent = 0


#         # uso percentual atual da memória RAM;
#         self.memory_usage_percent_inst.append(memory_usage_percent)
#         #uso percentual atual da memória da GPU (placa de vídeo);
#         self.gpu_memory_usage_percent_inst.append(gpu_memory_usage_percent)
#         #uso atual de memória RAM em gigabytes;
#         self.memory_usage_gigs_inst.append(memory_usage_gigs)
#         #uso atual de memória da GPU em gigabytes;
#         self.gpu_memory_usage_gigs_inst.append(gpu_memory_usage_gigs)
#         #porcentagem de memória RAM disponível;
#         self.memory_avail_percent_inst.append(memory_avail_percent)
#         #quantidade de memória RAM disponível em gigabytes;
#         self.memory_avail_gigs_inst.append(memory_avail_gigs)
#         #uso percentual atual da CPU;
#         self.cpu_usage_percent_inst.append(cpu_usage_percent)
#         #uso percentual atual da GPU (placa de vídeo).
#         self.gpu_usage_percent_inst.append(gpu_usage_percent)

#         #calcula a média de todas as informações acima;
#         self.memory_usage_percent_avg.append(sum(self.memory_usage_percent_inst) / self.ep_count)
#         self.gpu_memory_usage_percent_avg.append(
#             sum(self.gpu_memory_usage_percent_inst) / self.ep_count)
#         self.memory_usage_gigs_avg.append(sum(self.memory_usage_gigs_inst) / self.ep_count)
#         self.gpu_memory_usage_gigs_avg.append(sum(self.gpu_memory_usage_gigs_inst) / self.ep_count)
#         self.memory_avail_percent_avg.append(sum(self.memory_avail_percent_inst) / self.ep_count)
#         self.memory_avail_gigs_avg.append(sum(self.memory_avail_gigs_inst) / self.ep_count)
#         self.cpu_usage_percent_avg.append(sum(self.cpu_usage_percent_inst) / self.ep_count)
#         self.gpu_usage_percent_avg.append(sum(self.gpu_usage_percent_inst) / self.ep_count)

#         #Essa parte do código verifica se o episódio atual tem uma recompensa (ep_reward) melhor do que 
#         #a melhor recompensa registrada até agora (self.best_reward) e, se for o caso,
#         #atualiza a melhor recompensa (self.best_reward) 
#         #e o episódio em que ela ocorreu (self.best_reward_episode) para o episódio atual (self.ep_count). 
#         # Isso é útil para acompanhar o melhor desempenho alcançado pelo agente durante o treinamento.
#         if ep_reward > self.best_reward:
#             self.best_reward = ep_reward
#             self.best_reward_episode = self.ep_count


#         #Essa parte do código é responsável por monitorar a taxa de vitórias 
#         # do agente em um ambiente de aprendizado episódico. 
#         # A variável has_won é um booleano que indica se o agente venceu ou não o episódio atual. 
#         # Se has_won for verdadeiro, então o valor 1 é adicionado à lista ep_victories, caso contrário, o valor 0 é adicionado. 
#         # A lista ep_avg_victories é uma lista de média móvel que armazena a média das vitórias episódicas para cada episódio.
#         if self.is_episodic:
#             victory = 1 if has_won else 0
#             self.ep_victories.append(victory)
#             self.ep_avg_victories.append(sum(self.ep_victories) / self.ep_count)

#         #Essa parte inicializa o dicionário self.agent_info com as chaves presentes no dicionário agent_info, se self.agent_info for None (ou seja, não foi inicializado antes). Em seguida, 
#         #itera sobre as chaves em self.agent_info e define uma lista vazia como valor para cada uma dessas chaves.
#         if self.agent_info is None:
#             self.agent_info = dict.fromkeys(agent_info)
#             for key in self.agent_info:
#                 self.agent_info[key] = []

#         #Essa parte do método adiciona os valores dos dicionários do agent_info atual 
#         #para a lista de histórico self.agent_info. O loop percorre cada chave 
#         #do dicionário agent_info e adiciona o valor correspondente para a lista self.agent_info[key].
#         for key in agent_info:
#             self.agent_info[key].append(agent_info[key])

#         # batch episode calculation
#         #Essa parte do código verifica se o número de episódios coletados é o primeiro ou um múltiplo 
#         #de um parâmetro definido episode_batch_avg_calculation. 
#         # Se for, a média das recompensas dos últimos episode_batch_avg_calculation episódios é calculada 
#         # e adicionada à lista ep_avg_batch_rewards. 
#         # Se for o primeiro episódio, a recompensa do episódio atual é adicionada à lista ep_avg_batch_rewards. 
#         #A lista ep_avg_batch_rewards_episodes guarda os episódios em que a média das recompensas foi calculada.
#         if self.ep_count == 1 or self.ep_count % self.episode_batch_avg_calculation == 0:
#             if self.ep_count == 1:
#                 self.ep_avg_batch_rewards_episodes.append(self.ep_count)
#                 self.ep_avg_batch_rewards.append(ep_reward)
#             else:
#                 avg_rwd = sum(self.ep_rewards[(self.ep_avg_batch_rewards_episodes[-1] - 1):-1]) \
#                           / self.episode_batch_avg_calculation
#                 self.ep_avg_batch_rewards_episodes.append(self.ep_count)
#                 self.ep_avg_batch_rewards.append(avg_rwd)


class TestLoggerMethods (unittest.TestCase):
    def test_record_episode(self):
        env = GymEnv(id='FrozenLakeNotSlippery-v0')
        action_wrapper = env.get_action_wrapper()
        state_builder = FrozenLakeState()

        helper = ModelBuilder()
        helper.add_input_layer()
        helper.add_fullyconn_layer(256)
        helper.add_fullyconn_layer(256)
        helper.add_fullyconn_layer(256)
        helper.add_fullyconn_layer(256)
        helper.add_output_layer()

        dq_network = DDQNKeras(action_wrapper=action_wrapper, state_builder=state_builder,
                           learning_rate=0.005, gamma=0.90, use_memory=False,
                           per_episode_epsilon_decay=True, build_model=helper.get_model_layout())
        # Criar um objeto do tipo da classe que possui o método record_episode
        agent = GenericAgent(dq_network, FrozenlakeReward())

        # Definir valores de entrada para o método record_episode
        ep_reward = 100
        ep_actions = [0, 1, 2]
        has_won = True
        agent_info = {'param1': 0.5, 'param2': 'string'}
        steps_count = 100


        # Chamar o método record_episode com os valores de entrada definidos
        agent.record_episode(ep_reward, has_won, steps_count, agent_info, ep_actions)

        # Verificar se os valores de saída são iguais aos valores esperados
        self.assertEqual(agent.ep_count, 1)
        self.assertEqual(agent.ep_rewards, [100])
        self.assertEqual(agent.ep_avg_rewards, [100])
        self.assertEqual(agent.ep_agent_actions, [[0, 1, 2]])
        self.assertEqual(agent.avg_ep_agent_actions, [[0.0, 0.5, 1.0]])
        self.assertEqual(agent.ep_victories, [1])
        self.assertEqual(agent.ep_avg_victories, [1.0])
        self.assertEqual(agent.agent_info, {'param1': [0.5], 'param2': ['string']})
    

    # def setUp(self):
    #     self.logger = GymAgent("CartPole-v1")
    #     self.ep_count = 10
    #     self.play_rewards = [1, 2, 3, 4, 5]
    #     self.play_victories = 3
    #     self.num_matches = 5
    
    # def test_record_play_test(self):
    #     self.logger.record_play_test(self.ep_count, self.play_rewards, self.play_victories, self.num_matches)
        
    #     self.assertIn(self.ep_count, self.logger.play_ep_count)
    #     self.assertIn(self.num_matches, self.logger.play_match_count)
    #     self.assertEqual(self.play_victories / self.num_matches, self.logger.play_win_rates[-1])
    #     self.assertEqual(sum(self.play_rewards) / self.num_matches, self.logger.play_rewards_avg[-1])

if __name__ == '__main__':
    unittest.main()
