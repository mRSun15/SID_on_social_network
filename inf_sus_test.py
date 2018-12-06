import pickle

import csv
import seaborn as sns
from model_function import *
import pickle
import random

homo_degrees = ['strong']
# homo_degrees = ['weak','2_medium','1_medium','medium','medium_1','medium_2','medium_3','strong']
X_number = 6
X_list = []

for i in range(X_number):
    X_list.append([-1,1])


# plt.figure(num=1, figsize=(30, 10), dpi=80, facecolor='yellow', edgecolor='k')
# plt.figure(num=2, figsize=(20, 10), dpi=80, facecolor='yellow', edgecolor='k')
# plt.figure(num=3, figsize=(20, 10), dpi=80, facecolor='yellow', edgecolor='k')
# plt.figure(num=4, figsize=(30, 10), dpi=80, facecolor='yellow', edgecolor='k')

policies = ['thompson','greedy', 'e-greedy','balanced', 'ucb']


eps_s = [0.1,0.3,0.5,0.7,0.9]
count = 0

total_epochs = 1
print("X is:", X_list)
rank_list = {}
# state = True

graph = pickle.load(open("graph_data.pkl", 'rb'))

seed_nodes_paper = None
f = open('../inf-max-code-release/seed_set_IC/seed_set_11.txt')
for line in f.readlines():
    temp_line = line.strip('\n')
    if temp_line.startswith('Selected k SeedSet:'):
        temp_nodes = temp_line.split(':')[1].split()
        seed_nodes_paper = [int(n) for n in temp_nodes]
print(seed_nodes_paper)

seed_nodes_baseline = None
f = open('../inf-max-code-release/seed_set_IC/seed_set_10.txt')
for line in f.readlines():
    temp_line = line.strip('\n')
    if temp_line.startswith('Selected k SeedSet:'):
        temp_nodes = temp_line.split(':')[1].split()
        seed_nodes_baseline = [int(n) for n in temp_nodes]
print(seed_nodes_baseline)
seed = {'paper_inverse':seed_nodes_paper}


for (method_name, seed_nodes) in seed.items():
    overall_reward = {}
    overall_variance = {}
    overall_phase_adopted = {}
    overall_pair_vectors = []
    for policy in policies:
        if policy == 'e-greedy':
            for eps in eps_s:
                overall_reward[policy + str(eps)] = []
                overall_variance[policy + str(eps)] = []
                overall_phase_adopted[policy+str(eps)] = []
        else:
            overall_reward[policy] = []
            overall_variance[policy] = []
            overall_phase_adopted[policy] = []

    for homo_deg in homo_degrees:
        print('Homedeg is:', homo_deg)
        temp_pair_vectors = {}
        Graph,friend_pref, _= create_network(node_num=400,
                                               graph=graph,
                                               p=0.1,
                                               pref_range=10,
                                               pref_dim=2,
                                               homo_degree=homo_deg)

        adopt_nodes = {node: 0 for node in seed_nodes}

        eigen_central_vector = graph_eigen_centrality(Graph)
        temp_pair_vectors['network'] = eigen_central_vector

        temp_process_result = {}
        temp_mse_list = {}

        for policy in policies:
            if policy == 'e-greedy':
                for eps in eps_s:
                    temp_mse_list[policy + str(eps)] = {}
                    temp_process_result[policy + str(eps)] = {}
                    new_graph = copy.deepcopy(Graph)
                    new_adopted_node = copy.deepcopy(adopt_nodes)
                    new_friend_pref = copy.deepcopy(friend_pref)
                    # plt.subplot(161+count)
                    # count += 1
                    temp_reward, temp_reputation, temp_phase_adopted = plot_reward(individual_type='neutral',
                                                                                 policy=policy,
                                                                                 X_list = X_list,
                                                                                 Graph=new_graph,
                                                                                 friend_est_pref=new_friend_pref,
                                                                                 adopted_node=new_adopted_node,
                                                                                 eps_greedy=eps,
                                                                                 time_step=8,
                                                                                 process_result=temp_process_result[policy+str(eps)],
                                                                                 pref_mse_list=temp_mse_list[policy + str(eps)])

                    if len(temp_process_result[policy+str(eps)].keys()) == 0:
                        print("process_result is None, Error!")
                        exit(0)
                    overall_phase_adopted[policy+str(eps)].append(temp_phase_adopted)
                    overall_reward[policy+str(eps)].append(np.sum(list(temp_reward.values())))
                    temp_pair_vectors[policy + str(eps)] = np.array(list(temp_reward.values()))
                    overall_variance[policy+str(eps)].append(np.var([rep for k, rep in temp_reward.items()]))
                    # temp_rank_rep[policy+str(eps)]=np.var([rep for k, rep in temp_reputation.items()])
            else:
                temp_process_result[policy] = {}
                temp_mse_list[policy] = {}
                new_graph = copy.deepcopy(Graph)
                new_adopted_node = copy.deepcopy(adopt_nodes)
                new_friend_pref = copy.deepcopy(friend_pref)
                # plt.subplot(161+count)
                # count += 1
                temp_reward, temp_reputation, temp_phase_adopted = plot_reward(individual_type='neutral',
                                                             policy=policy,
                                                             X_list=X_list,
                                                             Graph=new_graph,
                                                             friend_est_pref=new_friend_pref,
                                                             adopted_node=new_adopted_node,
                                                             time_step=8,
                                                             process_result=temp_process_result[policy],
                                                             pref_mse_list=temp_mse_list[policy])
                overall_phase_adopted[policy].append(temp_phase_adopted)
                overall_reward[policy].append(np.sum(list(temp_reward.values())))
                overall_variance[policy].append(np.var([rep for k, rep in temp_reward.items()]))
                temp_pair_vectors[policy] = np.array(list(temp_reward.values()))
        overall_pair_vectors.append(temp_pair_vectors)

        sum_reward = {}
        std_reward = {}
        new_reward = overall_reward
        new_performance = {}
        # print(overall_reward['balanced'][0])
        for policy in policies:

            if policy == 'e-greedy':
                sum_reward[policy] = {}
                std_reward[policy] = {}

                for eps in eps_s:
                    # sum_reward[policy+str(eps)] = [np.sum([reward[count] for reward in overall_reward[policy+str(eps)]])/total_times
                    #                              for count in range(len(X_list))]
                    # new_reward[policy+str(eps)] = []

                    new_performance[policy+str(eps)] = {}
                    # for single_reward in overall_reward[policy+str(eps)]:
                    #     new_reward[policy+str(eps)].append(np.sum(single_reward))
                    new_performance[policy+str(eps)]['adopt_medium'] = np.median(new_reward[policy+str(eps)])
                    new_performance[policy+str(eps)]['adopt_mean'] = np.mean(new_reward[policy+str(eps)])
                    new_performance[policy+str(eps)]['adopt_std'] = np.std(new_reward[policy+str(eps)])
                    new_performance[policy+str(eps)]['var_mean'] = np.mean(overall_variance[policy+str(eps)])
                    new_performance[policy+str(eps)]['var_std'] = np.std(overall_variance[policy+str(eps)])
                    new_performance[policy+str(eps)]['var_medium'] = np.median(overall_variance[policy+str(eps)])
            else:
                # new_reward[policy] = []
                new_performance[policy] = {}
                # sum_reward[policy] = [np.sum([reward[count] for reward in overall_reward[policy]])/total_times for count in range(len(X_list))]
                # for single_reward in overall_reward[policy]:
                #     new_reward[policy].append(np.sum(single_reward))
                new_performance[policy]['adopt_mean'] = np.mean(new_reward[policy])
                new_performance[policy]['adopt_std'] = np.std(new_reward[policy])
                new_performance[policy]['adopt_medium'] = np.median(new_reward[policy])
                new_performance[policy]['var_mean'] = np.mean(overall_variance[policy])
                new_performance[policy]['var_std'] = np.std(overall_variance[policy])
                new_performance[policy]['var_medium'] = np.median(overall_variance[policy])


        with open('inf_sus_'+method_name+'/'+str(homo_deg)+ '_adopt_output.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in new_reward.items():
                writer.writerow([key, value])
        with open('inf_sus_'+method_name+'/'+str(homo_deg) + '_var_output.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in overall_variance.items():
                writer.writerow([key, value])
        with open('inf_sus_'+method_name+'/'+str(homo_deg) + '_phase_adopt.pkl', 'wb') as m_file:
            pickle.dump(overall_phase_adopted, m_file)

        with open('inf_sus_'+method_name+'/'+str(homo_deg) + '_pair_vectors.pkl', 'wb') as v_file:
            pickle.dump(overall_pair_vectors, v_file)
        # statistic, p_value = scipy.stats.ttest_ind(overall_reward['balanced'], overall_reward['greedy'])
        # print("Statistc: ", statistic, "; p_value: ", p_value)
        # print(new_performance)
        print('policies\t\treward_mean\treward_std\t\tvar_mean\tvar_std')
        for name, values in new_performance.items():
            print('{}\t\t{}\t{}\t\t{}\t{}'.format(name, values['adopt_medium'],values['adopt_std'], values['var_medium'], values['var_std']))

        # Plot the reward means
