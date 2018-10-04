import copy
from model_function import *
import scipy.stats
import csv
import seaborn as sns; sns.set()
import pickle

homo_degrees = ['weak', 'medium', 'strong']

X_number = 8
random_set = [-1, 1]
X_list = []
for i in range(X_number):
    X_list.append(np.random.choice(random_set,2, replace=True).tolist())

X = [[1, -1], [-1, 1], [-1, 1], [1, -1], [1, 1], [-1, -1], [-1, 1], [1, -1]]
policies = ['thompson','greedy', 'e-greedy','balanced', 'ucb']
eps_s = [0.1,0.3,0.5,0.7,0.9]
count = 0
total_times = 100
print("X is:", X_list)

for homo_deg in homo_degrees:
    overall_pair_vectors = []
    print('Homedeg is:', homo_deg)

    for time in range(total_times):
        temp_pair_vectors = {}
        print(time)
        temp_rank_rep = {}
        temp_process_result = {}
        temp_mse_list = {}
        Graph, friend_pref, adopt_nodes = create_network(node_num=200,
                                                         p=0.1,
                                                         pref_range=10,
                                                         pref_dim=2,
                                                         homo_degree=homo_deg)
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
                    temp_adoption, temp_reputation = plot_reward(individual_type='neutral',
                                                                 policy=policy,
                                                                 X_list = X_list,
                                                                 Graph=new_graph,
                                                                 friend_est_pref=new_friend_pref,
                                                                 adopted_node=new_adopted_node,
                                                                 eps_greedy=eps,
                                                                 time_step=10,
                                                                 process_result=temp_process_result[policy+str(eps)],
                                                                 pref_mse_list=temp_mse_list[policy + str(eps)])

                    if len(temp_process_result[policy+str(eps)].keys()) == 0:
                        print("process_result is None, Error!")
                        exit(0)
                    temp_pair_vectors[policy+str(eps)] = np.array(list(temp_reputation.values()))
            else:
                temp_process_result[policy] = {}
                temp_mse_list[policy] = {}
                new_graph = copy.deepcopy(Graph)
                new_adopted_node = copy.deepcopy(adopt_nodes)
                new_friend_pref = copy.deepcopy(friend_pref)
                # plt.subplot(161+count)
                # count += 1
                temp_adoption, temp_reputation = plot_reward(individual_type='neutral',
                                                             policy=policy,
                                                             X_list=X_list,
                                                             Graph=new_graph,
                                                             friend_est_pref=new_friend_pref,
                                                             adopted_node=new_adopted_node,
                                                             time_step=10,
                                                             process_result=temp_process_result[policy],
                                                             pref_mse_list=temp_mse_list[policy])
                temp_pair_vectors[policy] = np.array(list(temp_reputation.values()))
        overall_pair_vectors.append(temp_pair_vectors)


    with open('Correct_'+str(homo_deg) + '_pair_vectors.pkl', 'wb') as v_file:
        pickle.dump(overall_pair_vectors, v_file)
