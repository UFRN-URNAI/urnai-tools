from multiprocessing import Process
import os
from time import time

import GPUtil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from urnai.base.savable import Savable
from urnai.utils import constants
from urnai.utils.reporter import Reporter as rp


class Logger(Savable):
    """
    Logging class.
    Saves training parameters in python lists, which are pickled at saving,
    and also generates graphs based on those lists.
    """

    def __init__(self, ep_total, agent_name, model_name, model,
                 action_wrapper_name, agent_action_size, agent_action_names,
                 state_builder_name, reward_builder_name, env_name,
                 is_episodic=True, render=True, generate_bar_graphs_every=100, log_actions=True,
                 episode_batch_avg_calculation=10, rolling_avg_window_size=20,
                 graph_size_in_inches_width=12.8,
                 graph_size_in_inches_height=4.8,
                 threaded_saving=False):
        super().__init__(threaded_saving=threaded_saving)
        # Adding rolling avg size to pickle black list to allow us to regenerate graphs
        # with different rolling window sizes
        self.pickle_black_list.append('rolling_avg_window_size')

        # Training information
        self.agent_name = agent_name
        self.model_name = model_name
        self.model = model
        self.action_wrapper_name = action_wrapper_name
        self.state_builder_name = state_builder_name
        self.reward_builder_name = reward_builder_name
        self.env_name = env_name

        self.generate_bar_graphs_every = generate_bar_graphs_every

        self.log_actions = log_actions

        # Episode count
        self.ep_count = 0
        self.ep_total = ep_total

        # Reward count
        self.best_reward = -999999
        self.best_reward_episode = -1
        self.episode_batch_avg_calculation = episode_batch_avg_calculation
        self.rolling_avg_window_size = rolling_avg_window_size
        #contém a recompensa obtida em cada episódio executado
        self.ep_rewards = []
        #contém a média de recompensas obtidas em cada episódio
        self.ep_avg_rewards = []
        self.ep_avg_batch_rewards = []
        self.ep_avg_batch_rewards_episodes = []
        self.inside_training_test_avg_rwds = []

        # Steps count
        self.ep_steps_count = []
        self.ep_avg_steps = []

        # Win rate count
        self.ep_victories = []
        self.ep_avg_victories = []

        # Agent Action count
        self.agent_action_names = agent_action_names
        #número de ações tomadas pelo agente
        self.agent_action_size = agent_action_size
        self.ep_agent_actions = [[] for i in range(agent_action_size)]
        #lista utilizada para plotar um gráfico da evolução das ações tomadas pelo agente ao longo dos episódios.
        self.avg_ep_agent_actions = [[] for i in range(agent_action_size)]

        # Play testing count
        self.play_ep_count = []
        self.play_rewards_avg = []
        self.play_match_count = []
        self.play_win_rates = []

        # Some agent info
        self.agent_info = None

        self.is_episodic = is_episodic

        self.avg_reward_graph = None
        self.inst_reward_graph = None
        self.avg_steps_graph = None
        self.avg_winrate_graph = None

        # time and sps part
        self.training_start = time()
        self.episode_duration_list = []
        self.episode_sps_list = []
        self.avg_sps_list = []
        self.episode_temp_start_time = 0

        # performance report
        self.memory_usage_percent_inst = []
        self.memory_usage_gigs_inst = []
        self.memory_avail_percent_inst = []
        self.memory_avail_gigs_inst = []
        self.cpu_usage_percent_inst = []
        self.gpu_usage_percent_inst = []
        self.gpu_memory_usage_percent_inst = []
        self.gpu_memory_usage_gigs_inst = []
        self.memory_usage_percent_avg = []
        self.memory_usage_gigs_avg = []
        self.memory_avail_percent_avg = []
        self.memory_avail_gigs_avg = []
        self.cpu_usage_percent_avg = []
        self.gpu_usage_percent_avg = []
        self.gpu_memory_usage_percent_avg = []
        self.gpu_memory_usage_gigs_avg = []

        # Training report
        self.training_report = ''

        self.render = render
        self.graph_size_in_inches_width = graph_size_in_inches_width
        self.graph_size_in_inches_height = graph_size_in_inches_height

        self.log_training_start_information()

    # garante que todos os atributos de armazenamento de informações de treinamento sejam reiniciados, 
    # permitindo que um novo treinamento comece do zero.
    def reset(self):
        self.ep_count = 0 #define o número de episódios como zero

        self.ep_rewards = [] #limpa a lista de recompensas obtidas em cada episódio
        self.ep_avg_rewards = [] #limpa a lista de recompensas médias obtidas em cada episódio

        self.ep_steps_count = [] #limpa a lista de contagens de passos executados em cada episódio
        self.ep_avg_steps = [] #limpa a lista de contagens médias de passos  executados em cada episódio

        self.ep_victories = [] #limpa a lista do número de vitórias obtido em cada episódio
        self.ep_avg_victories = [] #limpa a lista do número médio de vitórias obtidos em cada episódio 

    # armazena a hora de início do episódio em questão.Isso é útil para calcular a duração de cada episódio de treinamento posteriormente.
    def record_episode_start(self):
        self.episode_temp_start_time = time() #variável que armazena o horário do início do episódio.

    def record_episode(self, ep_reward, has_won, steps_count, agent_info, ep_actions):
        #contador do número total de episódios.
        self.ep_count += 1


        #registra as ações tomadas pelo agente e calcula a média das ações tomadas até aquele ponto.
        for i in range(self.agent_action_size):

            #adiciona a ação do episódio atual
            self.ep_agent_actions[i].append(ep_actions[i]) 
            #calcula e adiciona a média das ações tomadas até este ponto.
            self.avg_ep_agent_actions[i].append(sum(self.ep_agent_actions[i]) / self.ep_count)


        #armazena a recompensa obtida no episódio atual
        self.ep_rewards.append(ep_reward)
        #calcula e armazena a média das recompensas obtidas até este episódio
        self.ep_avg_rewards.append(sum(self.ep_rewards) / self.ep_count)

        #adciona a contagem e a média da contagem de etapas que indica a quantidade de ações que o agente tomou durante um episódio.
        self.ep_steps_count.append(steps_count)
        self.ep_avg_steps.append(sum(self.ep_steps_count) / self.ep_count)

        # time and sps stuff
        #Essas métricas podem ajudar a avaliar o desempenho do agente ao longo do tempo e a identificar possíveis 
        # problemas ou melhorias que precisam ser feitas no modelo
        #Essa parte do método calcula e armazena as métricas relacionadas 
        # à duração e ao desempenho do agente durante o episódio.

        #mede quanto tempo o episódio levou para ser concluído (diferença entre o tempo atual e o tempo de início do ep).
        episode_duration = time() - self.episode_temp_start_time 
        #armazena a duração do episódio na lista que armazena a duração de cada episódio
        self.episode_duration_list.append(round(episode_duration, 1))

        if episode_duration != 0:
            #número de etapas do agente por segundo em cada episódio
            self.episode_sps_list.append(round(steps_count / episode_duration, 2))
        else:
            self.episode_sps_list.append(0)

        #armazena a média do número de etapas por segundo para todos os episódios concluídos até agora.        
        self.avg_sps_list.append(round(sum(self.episode_sps_list) / self.ep_count, 2))



        # performance stuff
        #uso da biblioteca psutil para coletar 
        # informações de uso do sistema em relação à CPU e à memória.

        #coleta a porcentagem de uso de memória atual do sistema.
        memory_usage_percent = psutil.virtual_memory().percent

        #calcula a porcentagem de memória disponível em relação ao total.
        memory_avail_percent = (psutil.virtual_memory().available * 100 /
                                psutil.virtual_memory().total)
        
        #calcula a quantidade de memória em uso em gigabytes.
        memory_usage_gigs = psutil.virtual_memory().used / 1024 ** 3
        #calcula a quantidade de memória disponível em gigabytes.
        memory_avail_gigs = psutil.virtual_memory().free / 1024 ** 3
        #coleta a porcentagem de uso da CPU no momento.
        cpu_usage_percent = psutil.cpu_percent()

        # has GPU available
        #Essa parte do código é responsável por tentar obter informações sobre a utilização da GPU (unidade de processamento gráfico) no sistema. 
        # Ele usa a biblioteca GPUtil para tentar acessar informações sobre a memória e a carga da GPU. 
        # Se conseguir acessar essas informações, ele armazena a porcentagem de uso de memória da GPU, o uso de memória da GPU em gigabytes e a porcentagem de uso da GPU. 
        # Caso contrário, ele define todos esses valores como 0.
        try:
            gpu_memory_usage_percent = GPUtil.getGPUs()[0].memoryUtil * 100
            gpu_memory_usage_gigs = GPUtil.getGPUs()[0].memoryUsed / 1000
            gpu_usage_percent = GPUtil.getGPUs()[0].load * 100

        except Exception:
            gpu_memory_usage_percent = 0
            gpu_memory_usage_gigs = 0
            gpu_usage_percent = 0


        # uso percentual atual da memória RAM;
        self.memory_usage_percent_inst.append(memory_usage_percent)
        #uso percentual atual da memória da GPU (placa de vídeo);
        self.gpu_memory_usage_percent_inst.append(gpu_memory_usage_percent)
        #uso atual de memória RAM em gigabytes;
        self.memory_usage_gigs_inst.append(memory_usage_gigs)
        #uso atual de memória da GPU em gigabytes;
        self.gpu_memory_usage_gigs_inst.append(gpu_memory_usage_gigs)
        #porcentagem de memória RAM disponível;
        self.memory_avail_percent_inst.append(memory_avail_percent)
        #quantidade de memória RAM disponível em gigabytes;
        self.memory_avail_gigs_inst.append(memory_avail_gigs)
        #uso percentual atual da CPU;
        self.cpu_usage_percent_inst.append(cpu_usage_percent)
        #uso percentual atual da GPU (placa de vídeo).
        self.gpu_usage_percent_inst.append(gpu_usage_percent)

        #calcula a média de todas as informações acima;
        self.memory_usage_percent_avg.append(sum(self.memory_usage_percent_inst) / self.ep_count)
        self.gpu_memory_usage_percent_avg.append(
            sum(self.gpu_memory_usage_percent_inst) / self.ep_count)
        self.memory_usage_gigs_avg.append(sum(self.memory_usage_gigs_inst) / self.ep_count)
        self.gpu_memory_usage_gigs_avg.append(sum(self.gpu_memory_usage_gigs_inst) / self.ep_count)
        self.memory_avail_percent_avg.append(sum(self.memory_avail_percent_inst) / self.ep_count)
        self.memory_avail_gigs_avg.append(sum(self.memory_avail_gigs_inst) / self.ep_count)
        self.cpu_usage_percent_avg.append(sum(self.cpu_usage_percent_inst) / self.ep_count)
        self.gpu_usage_percent_avg.append(sum(self.gpu_usage_percent_inst) / self.ep_count)

        #Essa parte do código verifica se o episódio atual tem uma recompensa (ep_reward) melhor do que 
        #a melhor recompensa registrada até agora (self.best_reward) e, se for o caso,
        #atualiza a melhor recompensa (self.best_reward) 
        #e o episódio em que ela ocorreu (self.best_reward_episode) para o episódio atual (self.ep_count). 
        # Isso é útil para acompanhar o melhor desempenho alcançado pelo agente durante o treinamento.
        if ep_reward > self.best_reward:
            self.best_reward = ep_reward
            self.best_reward_episode = self.ep_count


        #Essa parte do código é responsável por monitorar a taxa de vitórias 
        # do agente em um ambiente de aprendizado episódico. 
        # A variável has_won é um booleano que indica se o agente venceu ou não o episódio atual. 
        # Se has_won for verdadeiro, então o valor 1 é adicionado à lista ep_victories, caso contrário, o valor 0 é adicionado. 
        # A lista ep_avg_victories é uma lista de média móvel que armazena a média das vitórias episódicas para cada episódio.
        if self.is_episodic:
            victory = 1 if has_won else 0
            self.ep_victories.append(victory)
            self.ep_avg_victories.append(sum(self.ep_victories) / self.ep_count)

        #Essa parte inicializa o dicionário self.agent_info com as chaves presentes no dicionário agent_info, se self.agent_info for None (ou seja, não foi inicializado antes). Em seguida, 
        #itera sobre as chaves em self.agent_info e define uma lista vazia como valor para cada uma dessas chaves.
        if self.agent_info is None:
            self.agent_info = dict.fromkeys(agent_info)
            for key in self.agent_info:
                self.agent_info[key] = []

        #Essa parte do método adiciona os valores dos dicionários do agent_info atual 
        #para a lista de histórico self.agent_info. O loop percorre cada chave 
        #do dicionário agent_info e adiciona o valor correspondente para a lista self.agent_info[key].
        for key in agent_info:
            self.agent_info[key].append(agent_info[key])

        # batch episode calculation
        #Essa parte do código verifica se o número de episódios coletados é o primeiro ou um múltiplo 
        #de um parâmetro definido episode_batch_avg_calculation. 
        # Se for, a média das recompensas dos últimos episode_batch_avg_calculation episódios é calculada 
        # e adicionada à lista ep_avg_batch_rewards. 
        # Se for o primeiro episódio, a recompensa do episódio atual é adicionada à lista ep_avg_batch_rewards. 
        #A lista ep_avg_batch_rewards_episodes guarda os episódios em que a média das recompensas foi calculada.
        if self.ep_count == 1 or self.ep_count % self.episode_batch_avg_calculation == 0:
            if self.ep_count == 1:
                self.ep_avg_batch_rewards_episodes.append(self.ep_count)
                self.ep_avg_batch_rewards.append(ep_reward)
            else:
                avg_rwd = sum(self.ep_rewards[(self.ep_avg_batch_rewards_episodes[-1] - 1):-1]) \
                          / self.episode_batch_avg_calculation
                self.ep_avg_batch_rewards_episodes.append(self.ep_count)
                self.ep_avg_batch_rewards.append(avg_rwd)

    def record_play_test(self, ep_count, play_rewards, play_victories, num_matches):
        self.play_ep_count.append(ep_count)
        self.play_match_count.append(num_matches)
        self.play_win_rates.append(play_victories / num_matches)
        self.play_rewards_avg.append(sum(play_rewards) / num_matches)

    def log_training_start_information(self):
        text = ('\n   Agent: {}\n'.format(self.agent_name)
                + '   ActionWrapper: {}\n'.format(self.action_wrapper_name)
                + '   StateBuilder: {}\n'.format(self.state_builder_name)
                + '   RewardBuilder: {}\n'.format(self.reward_builder_name)
                + '   Environment: {}\n'.format(self.env_name)
                + '   Model: {}\n'.format(self.model_name))

        if hasattr(self.model, 'lib'):
            if self.model.neural_net_class is not None:
                if self.model.lib == constants.Libraries.KERAS:
                    stringlist = []
                    self.model.dnn.model.summary(print_fn=lambda x: stringlist.append(x))
                    short_model_summary = '\n'.join(stringlist)
                    text += '       ' + short_model_summary
                if self.model.lib == constants.Libraries.PYTORCH:
                    text += '       ' + self.model.dnn.model
            else:
                for idx, (layer) in enumerate(self.model.build_model):
                    text += '       Layer {}: {}\n'.format(idx, self.model.build_model[idx])
        else:
            for idx, (layer) in enumerate(self.model.build_model):
                text += '       Layer {}: {}\n'.format(idx, self.model.build_model[idx])

        self.training_report += text

        rp.report(text)

    def log_ep_stats(self):
        if self.ep_count > 0:

            agent_info = dict.fromkeys(self.agent_info)
            for key in agent_info:
                agent_info[key] = self.agent_info[key][-1]

            rp.report(
                ('Episode: {}/{} | Outcome: {} | Episode Avg. Reward: {:10.6f} | Episode Reward:'
                 ' {:10.6f} | Episode Steps: {:10.6f} | Best Reward was {} on episode: {} |'
                 'Episode Duration (seconds): {} | Episode SPS: {} | SPS AVG: {} | Agent info: {}')
                .format(self.ep_count,
                        self.ep_total,
                        self.ep_victories[-1],
                        self.ep_avg_rewards[-1],
                        self.ep_rewards[-1],
                        self.ep_steps_count[-1],
                        self.best_reward,
                        self.best_reward_episode,
                        self.episode_duration_list[-1],
                        self.episode_sps_list[-1],
                        self.avg_sps_list[-1], agent_info,
                        ),
            )
        else:
            rp.report('There are no recorded episodes!')

    def log_train_stats(self):
        if self.ep_count > 0:
            text = ('\n'
                    + 'Current Reward Avg.: {}'.format(sum(self.ep_rewards) / self.ep_count)
                    + ' Win rate: {:10.3f}%'.format((sum(self.ep_victories) / self.ep_count) * 100)
                    + ' Avg number of steps: {}'.format(sum(self.ep_avg_steps) / self.ep_count)
                    + ' Training Duration (seconds): {}'.format(
                        round(time() - self.training_start, 2))
                    + '\n')

            self.training_report += text

            rp.report(text)
        else:
            rp.report('There are no recorded episodes!')

    def plot_train_stats(self):
        self.plot_average_reward_graph()

        self.plot_average_steps_graph()

        if len(self.play_ep_count) > 0:
            self.plot_win_rate_percentage_over_play_testing_graph()
            self.plot_reward_average_over_play_testing_graph()

    def generalized_curve_plot(self, to_plot, label, title):
        return self.__plot_curve(range(self.ep_count), to_plot, 'Episode Count',
                                 label, title)

    def plot_average_reward_graph(self):
        # Plotting average reward graph
        return self.__plot_curve(range(self.ep_count), self.ep_avg_rewards, 'Episode Count',
                                 'Avg. Reward', r'Per Episode Avg. Reward')

    def plot_average_steps_graph(self):
        # Plotting average steps graph
        return self.__plot_curve(range(self.ep_count), self.ep_avg_steps, 'Episode Count',
                                 'Avg. Steps', r'Per Episode Avg. Steps')

    def plot_instant_reward_graph(self):
        # Plotting instant reward graph
        return self.__plot_curve(range(self.ep_count), self.ep_rewards, 'Episode Count',
                                 'Ep Reward', r'Per Episode Reward')

    def plot_batch_reward_graph(self):
        # Plotting average reward graph
        return self.__plot_curve(self.ep_avg_batch_rewards_episodes, self.ep_avg_batch_rewards,
                                 'Episode Count',
                                 'Ep Avg Reward', r'Average Reward Over Batch of Episodes')

    def plot_inside_training_test_avg_rwds(self):
        # Plotting average reward graph
        return self.__plot_curve(self.ep_avg_batch_rewards_episodes,
                                 self.inside_training_test_avg_rwds, 'Episode Count',
                                 'Training Tests Avg Reward',
                                 r'Training Tests Average Reward Evolution')

    def plot_win_rate_graph(self):
        # Plotting average win rate graph
        return self.__plot_curve(range(self.ep_count), self.ep_avg_victories, 'Episode Count',
                                 'Avg. Win Rate', r'Per Episode Avg. Win Rate')

    def plot_moving_avg_win_rate_graph(self):
        winrate_series = pd.Series(self.ep_victories)
        # creating a rolling window
        winrate_rolling_avg = winrate_series.rolling(window=self.rolling_avg_window_size)
        # getting the mean of the windows and dropping NaN (which will be the
        # first rolling_avg_window_size-1 elements)
        winrate_roll_dropped = winrate_rolling_avg.mean().dropna()

        return self.__plot_curve(range(self.rolling_avg_window_size - 1, self.ep_count),
                                 winrate_roll_dropped, 'Episode Count',
                                 'Rolling Avg. Win Rate',
                                 'Rolling Average Win Rate (window size: {})'.format(
                                     self.rolling_avg_window_size))

    def plot_moving_avg_reward_graph(self):
        reward_series = pd.Series(self.ep_rewards)
        # creating a rolling window
        reward_rolling_avg = reward_series.rolling(window=self.rolling_avg_window_size)
        # getting the mean of the windows and dropping NaN (which will be the first
        # rolling_avg_window_size-1 elements)
        reward_avg_dropped = reward_rolling_avg.mean().dropna()

        return self.__plot_curve(range(self.rolling_avg_window_size - 1, self.ep_count),
                                 reward_avg_dropped, 'Episode Count',
                                 'Rolling Avg. Reward',
                                 'Rolling Average Reward (window size: {})'.format(
                                     self.rolling_avg_window_size))

    def plot_win_rate_percentage_over_play_testing_graph(self):
        # Plotting win rate over play testing graph
        return self.__plot_bar(self.play_ep_count, [self.play_win_rates], ['Play'], 'Episode',
                               'Win rate (%)', 'Win rate percentage over play testing',
                               format_percent=True)

    def plot_reward_average_over_play_testing_graph(self):
        # Plotting reward average over play testing graph
        return self.__plot_bar(self.play_ep_count, [self.play_rewards_avg], ['Play'], 'Episode',
                               'Reward avg.', 'Reward avg. over play testing')

    def save_extra(self, persist_path):
        if not self.threaded_saving:
            self.unthreaded_save_extra(persist_path)
        else:
            self.threaded_save_extra(persist_path)

    def threaded_save_extra(self, persist_path):
        rp.report('THREADED Saving extra Logger object...',
                  verbosity_lvl=1)
        p = Process(target=self.unthreaded_save_extra, args=(persist_path,))
        p.start()

    def unthreaded_save_extra(self, persist_path):
        if self.avg_reward_graph is None or self.avg_steps_graph is None \
                or self.inst_reward_graph is None:
            self.render = False

            self.avg_reward_graph = self.plot_average_reward_graph()
            plt.savefig(
                persist_path + os.path.sep + self.get_default_save_stamp() + 'avg_reward_graph.png')
            plt.savefig(
                persist_path + os.path.sep + self.get_default_save_stamp() + 'avg_reward_graph.pdf')
            plt.close(self.avg_reward_graph)
            self.avg_reward_graph = None

            self.avg_steps_graph = self.plot_average_steps_graph()
            plt.savefig(
                persist_path + os.path.sep + self.get_default_save_stamp() + 'avg_steps_graph.png')
            plt.savefig(
                persist_path + os.path.sep + self.get_default_save_stamp() + 'avg_steps_graph.pdf')
            plt.close(self.avg_steps_graph)
            self.avg_steps_graph = None

            self.inst_reward_graph = self.plot_instant_reward_graph()
            plt.savefig(
                persist_path + os.path.sep + self.get_default_save_stamp()
                + 'inst_reward_graph.png')
            plt.savefig(
                persist_path + os.path.sep + self.get_default_save_stamp()
                + 'inst_reward_graph.pdf')
            plt.close(self.inst_reward_graph)
            self.inst_reward_graph = None

            self.avg_winrate_graph = self.plot_win_rate_graph()
            plt.savefig(
                persist_path + os.path.sep + self.get_default_save_stamp()
                + 'avg_winrate_graph.png')
            plt.savefig(
                persist_path + os.path.sep + self.get_default_save_stamp()
                + 'avg_winrate_graph.pdf')
            plt.close(self.avg_winrate_graph)
            self.avg_winrate_graph = None

            temp_fig = self.plot_moving_avg_win_rate_graph()
            plt.savefig(
                persist_path + os.path.sep + self.get_default_save_stamp()
                + 'rolling_avg_winrate_graph.png')
            plt.savefig(
                persist_path + os.path.sep + self.get_default_save_stamp()
                + 'rolling_avg_winrate_graph.pdf')
            plt.close(temp_fig)

            temp_fig = self.plot_moving_avg_reward_graph()
            plt.savefig(
                persist_path + os.path.sep + self.get_default_save_stamp()
                + 'rolling_avg_reward_graph.png')
            plt.savefig(
                persist_path + os.path.sep + self.get_default_save_stamp()
                + 'rolling_avg_reward_graph.pdf')
            plt.close(temp_fig)

            temp_fig = self.generalized_curve_plot(self.episode_duration_list,
                                                   'Episode Duration (Seconds)',
                                                   'Per Episode Duration In Seconds')
            plt.savefig(
                persist_path + os.path.sep + self.get_default_save_stamp()
                + 'ep_duration_graph.png')
            plt.savefig(
                persist_path + os.path.sep + self.get_default_save_stamp()
                + 'ep_duration_graph.pdf')
            plt.close(temp_fig)

            temp_fig = self.generalized_curve_plot(self.episode_sps_list, 'Episode SPS',
                                                   'Per Episode SPS')
            plt.savefig(
                persist_path + os.path.sep + self.get_default_save_stamp()
                + 'inst_ep_sps_graph.png')
            plt.savefig(
                persist_path + os.path.sep + self.get_default_save_stamp()
                + 'inst_ep_sps_graph.pdf')
            plt.close(temp_fig)

            temp_fig = self.generalized_curve_plot(self.avg_sps_list, 'Episode Avg. SPS',
                                                   'Per Episode Avg. SPS')
            plt.savefig(
                persist_path + os.path.sep + self.get_default_save_stamp() + 'avg_ep_sps_graph.png')
            plt.savefig(
                persist_path + os.path.sep + self.get_default_save_stamp() + 'avg_ep_sps_graph.pdf')
            plt.close(temp_fig)

            if self.log_actions:
                if self.agent_action_names is None:
                    self.agent_action_names = []
                    for i in range(self.agent_action_size):
                        self.agent_action_names.append('Action ' + str(i))
                # Plotting the rate of occurrence of each action in a different graph
                for i in range(self.agent_action_size):
                    if self.agent_action_names is not None:
                        # plot instant action usage graphs
                        action_graph = self.generalized_curve_plot(self.ep_agent_actions[i],
                                                                   self.agent_action_names[i],
                                                                   'Action ' +
                                                                   self.agent_action_names[
                                                                       i] + ' usage per episode.')
                        plt.savefig(
                            persist_path + os.path.sep + 'action_graphs' + os.path.sep
                            + 'instant' + os.path.sep +
                            self.agent_action_names[i] + '.png')
                        plt.close(action_graph)

                        action_graph = self.generalized_curve_plot(self.avg_ep_agent_actions[i],
                                                                   self.agent_action_names[i],
                                                                   'Action ' +
                                                                   self.agent_action_names[i]
                                                                   + ' average usage per episode.')
                        plt.savefig(
                            persist_path + os.path.sep + 'action_graphs' + os.path.sep
                            + 'average' + os.path.sep +
                            self.agent_action_names[i] + '.png')
                        plt.close(action_graph)

                # Plot action bars for each episode
                # First of all, transpose action usage list
                transposed = [list(x) for x in np.transpose(self.ep_agent_actions)]
                # Then, for each episode, get a bar graph showing each action usage
                for episode in range(self.ep_count):
                    if episode % self.generate_bar_graphs_every == 0:
                        values = transposed[episode]
                        bar_labels = self.agent_action_names
                        x_label = 'Actions'
                        y_label = 'How many times action was used'
                        title = 'Action usage at episode {}.'.format(episode)
                        bar_width = 0.2
                        action_graph = self.__plot_bar(values, bar_labels, x_label, y_label, title,
                                                       width=bar_width, horizontal=True)
                        plt.savefig(
                            persist_path + os.path.sep + 'action_graphs' + os.path.sep
                            + 'per_episode_bars' + os.path.sep + str(
                                episode) + '.png')
                        plt.close(action_graph)

                # Plotting the instant rate of occurrence of all actions in one single graph
                all_actions_graph = self.__plot_curves(range(self.ep_count), self.ep_agent_actions,
                                                       'Episode Count', 'Actions per Ep.',
                                                       self.agent_action_names,
                                                       'Instant rate of all actions')
                plt.savefig(
                    persist_path + os.path.sep + 'action_graphs' + os.path.sep + 'instant'
                    + os.path.sep + 'all_actions.png')
                plt.close(all_actions_graph)

                # Plotting the average rate of occurrence of all actions in one single graph
                all_actions_graph = self.__plot_curves(range(self.ep_count),
                                                       self.avg_ep_agent_actions, 'Episode Count',
                                                       'Actions avg. per Ep.',
                                                       self.agent_action_names,
                                                       'Average rate of all actions')
                plt.savefig(
                    persist_path + os.path.sep + 'action_graphs' + os.path.sep + 'average'
                    + os.path.sep + 'all_actions.png')
                plt.close(all_actions_graph)

            # Plotting agent info
            for key in self.agent_info:
                temp_fig = self.generalized_curve_plot(self.agent_info[key], 'Agent {}'.format(key),
                                                       'Per Episode Agent {} Data'.format(key))
                plt.savefig(
                    persist_path + os.path.sep + self.get_default_save_stamp()
                    + 'agent_{}_graph.png'.format(key))
                plt.savefig(
                    persist_path + os.path.sep + self.get_default_save_stamp()
                    + 'agent_{}_graph.pdf'.format(key))
                plt.close(temp_fig)

            # Plotting batch reward calculation graphs
            fig = self.plot_batch_reward_graph()
            plt.savefig(
                persist_path + os.path.sep + self.get_default_save_stamp()
                + 'batch_reward_graph.png')
            plt.savefig(
                persist_path + os.path.sep + self.get_default_save_stamp()
                + 'batch_reward_graph.pdf')
            plt.close(fig)
            fig = None

            # Plotting average reward for training tests
            if len(self.inside_training_test_avg_rwds) > 0:
                fig = self.plot_inside_training_test_avg_rwds()
                plt.savefig(
                    persist_path + os.path.sep + self.get_default_save_stamp()
                    + 'inside_training_test_avg_rwds.png')
                plt.savefig(
                    persist_path + os.path.sep + self.get_default_save_stamp()
                    + 'inside_training_test_avg_rwds.pdf')
                plt.close(fig)
                fig = None

            # Plotting performance info
            fig_names = [
                'memory_usage_percent_instant',
                'memory_usage_gigs_instant',
                'memory_avail_percent_instant',
                'memory_avail_gigs_instant',
                'gpu_memory_usage_percent_instant',
                'gpu_memory_usage_gigs_instant',
                'cpu_usage_percent_instant',
                'gpu_usage_percent_instant',
                'memory_usage_percent_average',
                'memory_usage_gigs_average',
                'memory_avail_percent_average',
                'memory_avail_gigs_average',
                'gpu_memory_usage_percent_average',
                'gpu_memory_usage_gigs_average',
                'cpu_usage_percent_average',
                'gpu_usage_percent_average',
            ]
            lst_to_save = [
                self.memory_usage_percent_inst,
                self.memory_usage_gigs_inst,
                self.memory_avail_percent_inst,
                self.memory_avail_gigs_inst,
                self.gpu_memory_usage_percent_inst,
                self.gpu_memory_usage_gigs_inst,
                self.cpu_usage_percent_inst,
                self.gpu_usage_percent_inst,
                self.memory_usage_percent_avg,
                self.memory_usage_gigs_avg,
                self.memory_avail_percent_avg,
                self.memory_avail_gigs_avg,
                self.gpu_memory_usage_percent_avg,
                self.gpu_memory_usage_gigs_avg,
                self.cpu_usage_percent_avg,
                self.gpu_usage_percent_avg,
            ]
            graph_titles = [
                'Instant Memory Usage (%)',
                'Instant Memory Usage (GB)',
                'Instant Memory Available (%)',
                'Instant Memory Available (GB)',
                'Instant GPU Memory Usage (%)',
                'Instant GPU Memory Usage (GB)',
                'Instant CPU usage (%)',
                'Instant GPU usage (%)',
                'Average Memory Usage (%)',
                'Average Memory Usage (GB)',
                'Average Memory Available (%)',
                'Average Memory Available (GB)',
                'Average GPU Memory Usage (%)',
                'Average GPU Memory Usage (GB)',
                'Average CPU usage (%)',
                'Average GPU usage (%)',
            ]
            for i in range(len(fig_names)):
                fig_name = fig_names[i]
                save_lst = lst_to_save[i]
                graph_title = graph_titles[i]

                temp_fig = self.generalized_curve_plot(save_lst, graph_title,
                                                       'Per episode ' + graph_title)
                plt.savefig(
                    persist_path + os.path.sep + 'performance_graphs' + os.path.sep
                    + self.get_default_save_stamp() + '{}.png'.format(fig_name))
                plt.savefig(
                    persist_path + os.path.sep + 'performance_graphs' + os.path.sep
                    + self.get_default_save_stamp() + '{}.pdf'.format(fig_name))
                plt.close(temp_fig)

            self.render = True

            with open(
                    persist_path
                    + os.path.sep
                    + self.get_default_save_stamp()
                    + 'overall_report.txt', 'w') as output:
                output.write(self.training_report)

    def __plot_curves(self, x, ys, x_label, y_label, y_labels, title):
        fig, ax = plt.subplots(figsize=self.__get_graph_size_in_inches())
        for i in range(len(ys)):
            ax.plot(x, ys[i], label=y_labels[i])

        ax.set(xlabel=x_label, ylabel=y_label, title=title)
        ax.grid()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, prop={'size': 6})

        if self.render:
            plt.ion()
            plt.show()
            plt.pause(0.001)

        return fig

    def __plot_curve(self, x, y, x_label, y_label, title):
        fig, ax = plt.subplots(figsize=self.__get_graph_size_in_inches())
        ax.plot(x, y)

        ax.set(xlabel=x_label, ylabel=y_label, title=title)
        ax.grid()

        if self.render:
            plt.ion()
            plt.show()
            plt.pause(0.001)

        return fig

    def __plot_bar(self, values, bar_labels, x_label, y_label, title, width=0.2, horizontal=False):

        if not horizontal:
            fig, ax = plt.subplots(figsize=self.__get_graph_size_in_inches())

            x = np.arange(len(bar_labels))  # List of label locations for the x values
            rects = ax.bar(x, values, width)

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(bar_labels)
            # Attach a text label above each bar in *rects*, displaying its height.
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords='offset points',
                            ha='center', va='bottom')

        else:
            width_in_inches, height_in_inches = self.__get_graph_size_in_inches()
            fig, ax = plt.subplots(figsize=(width_in_inches, height_in_inches + 5))

            x = np.arange(len(bar_labels))  # List of label locations for the x values
            rects = ax.barh(x, values, width)
            ax.set_xlabel(y_label)
            ax.set_ylabel(x_label)
            ax.set_title(title)
            ax.set_yticks(x)
            ax.set_yticklabels(bar_labels)

        if self.render:
            plt.ion()
            # plt.show()
            # plt.pause(0.001)

        return fig

    def __lerp(self, a, b, t):
        return (1 - t) * a + t * b

    def __get_graph_size_in_inches(self):
        return (self.graph_size_in_inches_width, self.graph_size_in_inches_height)
