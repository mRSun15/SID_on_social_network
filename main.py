import csv
import seaborn as sns
from model_function import *
import pickle
import random


sns.set()
# homo_degrees = ['weak', 'medium', 'strong']
homo_degrees = ['weak','2_medium','1_medium','medium','medium_1','medium_2','medium_3','strong']
X_number = 10
random_set = [-1, 1]
X_list = []
Product_list = [[-1,-1], [-1, 1], [1, -1], [1,1]]
# for i in range(X_number):
#     X_list.append(np.random.choice(random_set,2, replace=True).tolist())


plt.figure(num=1, figsize=(30, 10), dpi=80, facecolor='yellow', edgecolor='k')
plt.figure(num=2, figsize=(20, 10), dpi=80, facecolor='yellow', edgecolor='k')
# plt.figure(num=3, figsize=(20, 10), dpi=80, facecolor='yellow', edgecolor='k')
# plt.figure(num=4, figsize=(30, 10), dpi=80, facecolor='yellow', edgecolor='k')

policies = ['thompson','greedy', 'e-greedy','balanced', 'ucb']

eps_s = [0.1,0.3,0.5,0.7,0.9]
count = 0
overall_reward = {}
overall_variance = {}
total_epochs = 1
print("X is:", X_list)
rank_list = {}
# state = True
Graph_list = {}
for homo_deg in homo_degrees:
    overall_phase_adopted = {}
    overall_pair_vectors = []
    overall_processed_res = []
    overall_mse_list = []
    rank_reputation = []
    relative_reputation = []
    overall_network_pref = []
    print('Homedeg is:', homo_deg)

    for policy in policies:
        if policy == 'e-greedy':
            for eps in eps_s:
                overall_reward[policy+str(eps)] = []
                overall_variance[policy+str(eps)] = []
                overall_phase_adopted[policy+str(eps)] = []
        else:
            overall_reward[policy] = []
            overall_variance[policy] = []
            overall_phase_adopted[policy] = []
    for epoch in range(total_epochs):
        if epoch not in Graph_list:
            Graph, friend_pref, adopt_nodes = create_network(node_num=400,
                                                             p=0.1,
                                                             pref_range=10,
                                                             pref_dim=2,
                                                             homo_degree=homo_deg)
            Graph_list[epoch] = copy.deepcopy(Graph)
        else:
            Graph,friend_pref, adopt_nodes= create_network(node_num=400,
                                                           graph=Graph_list[epoch],
                                                           p=0.1,
                                                           pref_range=10,
                                                           pref_dim=2,
                                                           homo_degree=homo_deg)
        Products = {"positive": [], "negative": []}
        for product in Product_list:
            if Product_Property_Catch(Graph, product) > 0:
                Products['positive'].append(product)
            else:
                Products['negative'].append(product)

        print(Products)
        pos_prods = np.random.choice(np.arange(len(Products['positive'])), int(X_number*0.3), replace=True).tolist()
        pos_prods = [Products['positive'][i] for i in pos_prods]
        neg_prods = np.random.choice(np.arange(len(Products['negative'])), int(X_number*0.7), replace=True).tolist()
        neg_prods = [Products['negative'][i] for i in neg_prods]

        X_list = pos_prods + neg_prods
        print(X_list)
        random.shuffle(X_list)
        print(X_list)

        for time in range(10):
            print(time)
            temp_pair_vectors = {}
            temp_rank_rep = {}
            temp_process_result = {}
            temp_mse_list = {}
            overall_network_pref.append(Network_Pref_Cal(Graph, X_list))
            # plot the initial state
            # plt.subplot(161+count)
            # count += 1
            # Graph_initial_check(Graph=Graph,
            #                     X=X,
            #                     adopted_node=adopt_nodes,
            #                     homo_degree=homo_deg)
            eigen_central_vector = graph_eigen_centrality(Graph)
            temp_pair_vectors['network'] = eigen_central_vector
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
                        temp_pair_vectors[policy+str(eps)] = np.array(list(temp_reward.values()))
                        overall_reward[policy+str(eps)].append(np.sum(list(temp_reward.values())))
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
                    temp_rank_rep[policy] = np.var([rep for k, rep in temp_reward.items()])
                    temp_pair_vectors[policy] = np.array(list(temp_reward.values()))
            overall_pair_vectors.append(temp_pair_vectors)
            overall_processed_res.append(temp_process_result)
            overall_mse_list.append(temp_mse_list)

        # compute the every time's ranking

        # rep_items = temp_rank_rep.items()
        # back_rep_items = [[v[1], v[0]] for v in rep_items]
        # back_rep_items.sort()
        # temp_rela_rep = {}
        # optimal_relative = back_rep_items[0][0]
        # for rank in range(0, len(back_rep_items)):
        #     temp_rank_rep[back_rep_items[rank][1]] = rank+1
        #     temp_rela_rep[back_rep_items[rank][1]] = (back_rep_items[rank][0] - optimal_relative)/(optimal_relative+10e-10)
        # rank_reputation.append([temp_rank_rep[key] for key in policies_without_e])
        # relative_reputation.append([temp_rela_rep[key] for key in policies_without_e])

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


    with open('test_data_neg/'+str(homo_deg)+ '_adopt_output.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in new_reward.items():
            writer.writerow([key, value])
    with open('test_data_neg/'+str(homo_deg) + '_var_output.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in overall_variance.items():
            writer.writerow([key, value])
    with open('test_data_neg/'+str(homo_deg) + '_process_res.pkl', 'wb') as p_file:
        pickle.dump(overall_processed_res, p_file)
    with open('test_data_neg/'+str(homo_deg) + '_pair_vectors.pkl', 'wb') as v_file:
        pickle.dump(overall_pair_vectors, v_file)
    with open('test_data_neg/'+str(homo_deg) + '_mse_lists.pkl', 'wb') as m_file:
        pickle.dump(overall_mse_list, m_file)
    with open('test_data_neg/'+str(homo_deg) + '_phase_adopt.pkl', 'wb') as m_file:
        pickle.dump(overall_phase_adopted, m_file)
    with open('test_data_neg/'+str(homo_deg) + '_network_pref.pkl', 'wb') as m_file:
        pickle.dump(overall_network_pref, m_file)


    # statistic, p_value = scipy.stats.ttest_ind(overall_reward['balanced'], overall_reward['greedy'])
    # print("Statistc: ", statistic, "; p_value: ", p_value)
    # print(new_performance)
    print('policies\t\treward_mean\treward_std\t\tvar_mean\tvar_std')
    for name, values in new_performance.items():
        print('{}\t\t{}\t{}\t\t{}\t{}'.format(name, values['adopt_medium'],values['adopt_std'], values['var_medium'], values['var_std']))

    # Plot the reward means

    means = [new_performance[policy]['adopt_mean'] for policy in new_performance.keys()]
    median = [new_performance[policy]['adopt_medium']  for policy in new_performance.keys()]
    std = [new_performance[policy]['adopt_std'] for policy in new_performance.keys()]
    ind = np.arange(len(new_performance.keys()))
    width = 0.6
    plt.figure(1)
    plt.subplot(331 + count)
    p1 = plt.bar(ind, median, width, yerr=std)

    plt.xticks(ind, new_performance.keys())
    plt.yticks(np.arange(0, 80, 10),np.arange(60, 140, 10))



    # Plot the HeatMap

    # plt.figure(2)
    # plt.subplot(131 + count)
    # rank_rep_np = np.array(rank_reputation)
    # x_labels = policies_without_e
    # sns.heatmap(rank_rep_np,1,len(x_labels)+1, xticklabels=x_labels)
    # plt.figure(3)
    # plt.subplot(131 + count)
    # rela_rep_np = np.array(relative_reputation)
    # max_rela = np.max(rela_rep_np)
    # sns.heatmap(rela_rep_np,0,4, xticklabels=x_labels)



    # Plot the reputation variance

    means = [new_performance[policy]['var_mean'] for policy in new_performance.keys()]
    median = [new_performance[policy]['var_medium'] for policy in new_performance.keys()]
    std = [new_performance[policy]['var_std'] for policy in new_performance.keys()]
    ind = np.arange(len(new_performance.keys()))
    plt.figure(2)
    plt.subplot(331+count)
    p1 = plt.bar(ind, median, width, yerr=std)
    plt.xticks(ind, new_performance.keys())
    plt.yticks(np.arange(0, 20, 10))
    count += 1
plt.show()


# total_times = 30
# for X in X_list:
#     print("X is:", X)
#     for homo_deg in homo_degrees:
#         for time in range(total_times):
#             Graph, friend_pref, adopt_nodes = create_network(node_num=50,
#                                                              p=0.1,
#                                                              pref_range=10,
#                                                              pref_dim=2,
#                                                              homo_degree=homo_deg)
#
#             for policy in policies:
#                 new_graph = copy.deepcopy(Graph)
#                 new_adopted_node = copy.deepcopy(adopt_nodes)
#                 # plt.subplot(161+count)
#                 count += 1
#                 if policy not in overall_adoption:
#                     overall_adoption[policy] = plot_adoption(individual_type='neutral',
#                                                              policy=policy,
#                                                              X = X,
#                                                              Graph=new_graph,
#                                                              friend_est_pref=friend_pref,
#                                                              adopted_node=new_adopted_node)
#                 else:
#                     overall_adoption[policy] += plot_adoption(individual_type='neutral',
#                                                              policy=policy,
#                                                              X = X,
#                                                              Graph=new_graph,
#                                                              friend_est_pref=friend_pref,
#                                                              adopted_node=new_adopted_node)
#         for policy in policies:
#             overall_adoption[policy] /= total_times
#         print('homophily is :', homo_deg)
#         print(overall_adoption)
