import copy

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from utils import *


def update_friend_pref(prior_theta, X, y, w_ij):
    mu_t = np.mat(prior_theta['mean']).reshape((-1, 1))
    var_t = np.mat(prior_theta['cov'])
    dim = len(X)
    X = np.mat(X)
    S_t = var_t
    inv_S_t = np.linalg.pinv(S_t)
    inv_S_t_1 = X.T * X / w_ij + inv_S_t
    S_t_1 = np.linalg.pinv(inv_S_t_1)
    mu_t_1 = S_t_1 * (X.T * y / w_ij + inv_S_t * mu_t)
    mu_t_1 = np.asarray(mu_t_1.T)
    return np.squeeze(mu_t_1.T).astype(float), np.array(S_t_1).astype(float)
    # return [[mu_t_1[i,0], S_t_1[i,i]] for i in range(dim)]

def balanced_strategy(node_list):
    '''

    :param node_list: the list of all available nodes
    :return: the chosen node list

    '''
    node_num = len(node_list)
    random_chose = np.random.choice([True, False], node_num, replace=True)

    chosen_nodes = [node_list[i] for i in range(node_num) if random_chose[i] == True]

    return  chosen_nodes

def greedy_strategy(reward_list):
    '''

    :param reward_list: the reward list for all the available nodes
    :return: the chosen node list
    '''
    result_nodes = []
    for node in reward_list.keys():
        if reward_list[node] >= 0:
            result_nodes.append(node)

    return result_nodes

def e_greedy_strategy(reward_list, epsilon_greedy):
    '''

    :param reward_list: the reward list for all the availabel nodes
    :param epsilon_greedy: the probability for chosen greedy
    :return: the chosen node list
    '''
    result_nodes = []
    for node in reward_list.keys():
        if np.random.uniform(0, 1, 1) <= epsilon_greedy:
            if reward_list[node] >= 0:
                result_nodes.append(node)
        else:
            if np.random.uniform(0, 1, 1) <= 0.5:
                result_nodes.append(node)

    return  result_nodes

def thompson_sample(Graph, individual_node, available_nodes, est_friend_pred, new_reward, X, sample_times = 20):
    for nbr in available_nodes:
        estimate_mean = est_friend_pred[nbr]['mean']
        estimate_cov = est_friend_pred[nbr]['cov']
        new_reward[nbr] = 0
        try:
            estimate_sample = np.random.multivariate_normal(mean=estimate_mean,
                                                            cov=estimate_cov,
                                                            size=sample_times)
        except Exception as e:
            print(e)
            print(estimate_mean.dtype)
            print(estimate_cov.dtype)
            exit()
        for times in range(sample_times):
            new_reward[nbr] += logistic_func(X, estimate_sample[times])

    result_nodes = [key for key, value in new_reward.items() if value > 0]

    return result_nodes

def ucb_strategy(Graph, individual_node ,ucb_counts, new_reward):

    result_nodes = []
    for nbr in new_reward.keys():
        past_chosen = ucb_counts[individual_node][nbr]
        total_counts = past_chosen['choose'] + past_chosen['not_choose']
        if (new_reward[nbr] + np.sqrt(2 * np.log1p(total_counts) / (past_chosen['choose'] + 1))) >\
                (0 + np.sqrt(2 * np.log1p(total_counts) / (past_chosen['not_choose'] + 1))):

            result_nodes.append(nbr)
            ucb_counts[individual_node][nbr]['choose'] += 1
        else:
            ucb_counts[individual_node][nbr]['not_choose'] += 1
    return result_nodes

def friend_selection(Graph, individual_type, individual_node, policy, friend_pref, past_reward, X, time, parent_nodes,
                    ucb_counts,epsilon_greedy=0.5):
    '''

    :param Graph:  networkx.graph
    :param individual_type: 'neutral' or 'risk-aversion' (different type for individual)
    :param individual_node: active node, person who will recommend products to his neighbors
    :param policy: different strategy/policy, e.g "thompson", "greedy", "balanced"...
    :param friend_pref: predicted_friend's preference.
    :param past_reward: dict, the total reward from time 0.....T-1
    :param X: product's attribute
    :param time: current time T
    :param parent_nodes: the nodes who recommend to active node(individual node)
    :param ucb_counts: utilized by ucb
    :param epsilon_greedy: decide the probability of choosing greedy strategy or balanced.
    :return:

    Attention:
        the function has been rewritten to the new version, which is more similar to the Independent Cascade model.

    '''
    new_reward = {}
    # available neighbors, exclude the pareent_nodes.
    availabel_nodes = [node for node in Graph.neighbors(individual_node)]
    availabel_nodes = list(set(availabel_nodes) - set(parent_nodes[individual_node]))
    if len(availabel_nodes) == 0:
        # isolated node
        return [], []

    # compute the predicted reward for every node.

    if individual_type == 'neutral':
        for nbr in availabel_nodes:
            if nbr in friend_pref[time - 1][individual_node]:
                new_reward[nbr] = estimate_reward(X, friend_pref[time - 1][individual_node][nbr])
            elif nbr in friend_pref[time - 2][individual_node]:
                friend_pref[time - 1][individual_node][nbr] = friend_pref[time - 2][individual_node][nbr]
                new_reward[nbr] = estimate_reward(X, friend_pref[time - 2][individual_node][nbr])
            else:
                new_reward[nbr] = estimate_reward(X, friend_pref[0][individual_node][nbr])
    elif individual_type == 'risk':
        for nbr in availabel_nodes:
            if nbr in friend_pref[time - 1][individual_node]:
                new_reward[nbr] = estimate_risk_adverse(X, friend_pref[time - 1][individual_node][nbr])
            elif nbr in friend_pref[time - 2][individual_node]:
                friend_pref[time - 1][individual_node][nbr] = friend_pref[time - 2][individual_node][nbr]
                new_reward[nbr] = estimate_risk_adverse(X, friend_pref[time - 2][individual_node][nbr])
            else:
                new_reward[nbr] = estimate_risk_adverse(X, friend_pref[0][individual_node][nbr])

    # chose the friend according to the policy
    # the selected results is a list, including all the possible nodes.
    chosen_nodes = []

    if policy == 'balanced':
        chosen_nodes = balanced_strategy(availabel_nodes)
    elif policy == 'greedy':
        chosen_nodes = greedy_strategy(new_reward)
    elif policy == 'e-greedy':
        chosen_nodes = e_greedy_strategy(new_reward, epsilon_greedy)
    elif policy == 'thompson':
        chosen_nodes = thompson_sample(Graph, individual_node, availabel_nodes, friend_pref[time - 1][individual_node], new_reward, X, 10)
    elif policy == 'ucb':
        chosen_nodes = ucb_strategy(Graph, individual_node, ucb_counts, new_reward)

    if individual_type == 'risk' or policy == 'ucb' or policy == 'thompson':
        for chosen_node in chosen_nodes:
            new_reward[chosen_node] = estimate_reward(X, friend_pref[time - 1][individual_node][chosen_node])
    # print(friend_pref[time-1][individual_node][nbr]['mean'])
    # print(new_reward[chosen_node])

    if len(chosen_nodes) < 0:
        return [], []

    return chosen_nodes, [new_reward[chosen_node] for chosen_node in chosen_nodes]


def Union_Graph(G, H):
    mapping = {}
    connected_edge_num = 3
    G_node_num = len(G.nodes())
    for node in H.nodes():
        mapping[node] = node+G_node_num
    H = nx.relabel_nodes(H, mapping)

    U = nx.union(G, H)
    for edge_id in range(connected_edge_num):
        G_node = np.random.choice(G.nodes())
        H_node = np.random.choice(H.nodes())
        U.add_edge(G_node, H_node)
    return U

def create_network(node_num, p, pref_range, pref_dim=2, homo_degree='strong', graph_type='bba', mean_pref=5, std_pref=6):
    friend_pref_estimated = {}
    friend_pref_estimated[0] = {}

    if graph_type == 'erdo':
        Graph = nx.erdos_renyi_graph(node_num, p)
    elif graph_type == 'ws':
        Graph = nx.watts_strogatz_graph(n=node_num,
                                        k=int(0.3 * node_num),
                                        p=p)
    elif graph_type == 'bba':
        G1_num = int(0.2*node_num)
        G1 = nx.barabasi_albert_graph(n = G1_num,
                                      m = int(G1_num*0.3))
        G2_num = int(0.3*node_num)
        G2 = nx.barabasi_albert_graph(n = G2_num,
                                      m = int(G2_num*0.3))
        G3_num = node_num - G1_num - G2_num
        G3 = nx.barabasi_albert_graph(n = G3_num,
                                      m = int(G3_num*0.3))
        G1 = Union_Graph(G1, G2)
        Graph = Union_Graph(G1, G3)

    Graph = set_personal_pref(Graph, homophily_degree=homo_degree, pref_dim=pref_dim)

    edges = Graph.edges()

    for edge in edges:
        # edge_weight = np.random.normal(mean_pref, std_pref)
        edge_weight = 1
        edge_weight = np.abs(edge_weight)

        Graph.add_edge(edge[0], edge[1], weight=edge_weight)
        Graph.add_edge(edge[1], edge[0], weight=edge_weight)

    node_pref = nx.get_node_attributes(Graph, 'pref')
    edge_strength = nx.get_edge_attributes(Graph, 'weight')

    for node in Graph.nodes():
        friend_pref_estimated[0][node] = {}
        for nbr in Graph[node]:
            try:
                friend_pref_estimated[0][node][nbr] = {}
                pref_mean = [node_pref[node][dim] for dim in range(pref_dim)]
                temp_cov = 0.5*np.identity(pref_dim)
                pref_noise = np.random.multivariate_normal(mean=np.zeros(pref_dim),
                                                           cov=temp_cov)

                # pref_mean = np.array(pref_mean)+pref_noise
                pref_mean = np.array(pref_mean)
                if (node, nbr) in edge_strength:
                    pref_cov = [1 / edge_strength[(node, nbr)] for dim in range(pref_dim)]
                    pref_cov = np.diag(np.array(pref_cov)).astype(float)
                else:
                    pref_cov = [1 / edge_strength[(nbr, node)] for dim in range(pref_dim)]
                    pref_cov = np.diag(np.array(pref_cov)).astype(float)
                friend_pref_estimated[0][node][nbr]['mean'] = pref_mean.astype(float)
                friend_pref_estimated[0][node][nbr]['cov'] = pref_cov

            except Exception as e:
                print(e)

    seed_node_num = int(node_num * 0.02)
    seed_nodes = np.random.choice(node_num, seed_node_num, replace=False)

    adopted_node = {node: [0] for node in seed_nodes}

    # print("successfully create graph")
    return Graph, friend_pref_estimated, adopted_node


def set_personal_pref(Graph, homophily_degree, pref_dim):
    # print("set personal pref")
    b_combinations_ = np.random.multivariate_normal(mean=np.zeros((pref_dim)),
                                                    cov=np.identity(pref_dim),
                                                    size=2).reshape(pref_dim, 2)
    b_sum = np.sum(np.abs(b_combinations_), axis=1, keepdims=True)
    b_combinations = b_combinations_ / b_sum
    eigenvals, eigenvects = graph_eigen(Graph)

    # print(eigenvects[:,:2].T)
    # print("--------------------------------")
    # print(eigenvects[:,-3:-1].T)
    if homophily_degree == 'strong':
        # 1~5
        b_combinations = b_combinations.dot(eigenvects[:, 0:2].T).T
        # print(b_combinations)
        # new_b_combinations = b_combinations_.dot(eigenvects[:,-3:-1].T).T
        # print(new_b_combinations)
    elif homophily_degree == 'medium':
        b_combinations = b_combinations.dot(eigenvects[:, 15:17].T).T
    elif homophily_degree == 'weak':
        b_combinations = b_combinations.dot(eigenvects[:, 40:42].T).T
        # print(b_combinations)

    for n_idx, n in enumerate(Graph.nodes()):
        Graph.node[n]['pref'] = b_combinations[n_idx]

    # node_pref = nx.get_node_attributes(Graph, 'pref')
    # for dim in range(pref_dim):
    #     pref_list = [node_pref[node][dim] for node in Graph.nodes()]
    #     pref_std = np.std(pref_list)
    #     pref_mean = np.mean(pref_list)
    #
    #     for n in Graph.nodes():
    #         Graph.node[n]['pref'][dim] = (Graph.node[n]['pref'][dim] - pref_mean) / pref_std

    return Graph


def preference_update(friend_feedback, individual_node, Graph, friend, epsilon=0.05):
    if friend_feedback == 1:
        friend_personal_pref = Graph.node[friend]['pref']
        individual_personal_pref = Graph.node[individual_node]['pref']
        try:
            updated_friend_pref = (1-epsilon) * friend_personal_pref + epsilon * individual_personal_pref
        except Exception as e:
            print(e)
            print(epsilon * friend_personal_pref)
            print((1 - epsilon) * individual_personal_pref)
            print(friend_personal_pref)
            print(individual_personal_pref)
        Graph.node[friend]['pref'] = updated_friend_pref
    return Graph

def compute_pref_mse(estimate_pref, Graph):
    mse_loss = 0
    for node in Graph.nodes():
        for neighbor in Graph[node]:
            mse_loss += mean_squared_error(estimate_pref[node][neighbor]['mean'], Graph.node[neighbor]['pref'])

    return mse_loss

def graph_simulation(graph, individual_type, policy, friend_pref, adopted_node,
                     X_list, total_time, eps_greedy, process_result, pref_mse_list):

    total_past_reward = {}
    collective_adopted = []
    reputation = {}
    ucb_counts = {}
    for n in graph.nodes():
        ucb_counts[n] = {}
        reputation[n] = 0
        for nbr in graph.neighbors(n):
            # print(n, nbr)
            ucb_counts[n][nbr] = {'choose': 0, 'not_choose': 0}
    name_count = 1
    pref_mse_list['0_0_0'] = compute_pref_mse(friend_pref[0], graph)



    for X in X_list:
        # print("Current X is:", X)
        temp_adopted_nodes = copy.deepcopy(adopted_node)
        temp_remove_nodes = {}
        parent_recommends = {}
        for n in graph.nodes():
            parent_recommends[n] = []
        last_time = total_time-1
        X_name = ' '.join(str(e) for e in X) + '_' + str(name_count)
        name_count += 1
        total_past_reward[X_name] = {}

        for time in range(1, total_time):
            # print(time)
            # print('time is: ', time)
            flag = False
            cur_reward = {}
            cur_friend_pref = {}
            if len(temp_adopted_nodes) <= 0:
                last_time = time - 1
                # print("No adopted nodes, error!")
                # print(last_time)
                break
            # print('For time: ',time, "Current adopt node: ",list(temp_adopted_nodes.keys()))
            new_adopted_nodes = {}
            for node in graph.nodes():

                if node in temp_adopted_nodes and len(graph.neighbors(node)) > 0:
                    # adopted recommendation stratedy
                    chosen_list, reward = friend_selection(Graph=graph,
                                                           individual_type=individual_type,
                                                           individual_node=node,
                                                           friend_pref=friend_pref,
                                                           X=X,
                                                           time=time,
                                                           policy=policy,
                                                           past_reward=total_past_reward[X_name],
                                                           parent_nodes=parent_recommends,
                                                           ucb_counts=ucb_counts,
                                                           epsilon_greedy=eps_greedy)


                    cur_friend_pref[node] = {}
                    if len(chosen_list) <= 0:
                        # print("Node: ",node," choose no node!")
                        for nbr in graph[node]:
                            cur_friend_pref[node][nbr] = friend_pref[time - 1][node][nbr]
                        continue
                    friend_reward = {}
                    for chosen_node in chosen_list:
                        friend_reward[chosen_node] = logistic_func(X, graph.node[chosen_node]['pref'])
                    cur_reward[node] = [chosen_list, np.sum(list(friend_reward.values()))]
                    # print("Time: ", time," Current reward is ", friend_reward)

                    # update the predicted preference

                    for chosen_node in chosen_list:
                        if friend_reward[chosen_node] >= 0:
                            # print('recommendation successful!')
                            node_feedback = 1
                            reputation[node] += 1
                            if (chosen_node not in temp_adopted_nodes) and (chosen_node not in temp_remove_nodes):
                                new_adopted_nodes[chosen_node] = [time]

                            parent_recommends[chosen_node].append(node)
                        else:
                            node_feedback = 0
                        flag = True

                        new_mean, new_cov = update_friend_pref(prior_theta=friend_pref[time - 1][node][chosen_node],
                                                               X=X,
                                                               y=node_feedback,
                                                               w_ij=graph[node][chosen_node]['weight'])
                        cur_friend_pref[node][chosen_node] = {}
                        cur_friend_pref[node][chosen_node]['mean'] = new_mean
                        cur_friend_pref[node][chosen_node]['cov'] = new_cov
                        # graph = preference_update(Graph=graph,
                        #                           friend_feedback=node_feedback,
                        #                           individual_node=node,
                        #                           friend=chosen_node,
                        #                           epsilon=0.05)
                    for nbr in graph[node]:
                        if nbr not in chosen_list:
                            cur_friend_pref[node][nbr] = friend_pref[time - 1][node][nbr]

                    # del temp_adopted_nodes[node]
                    # print("delete: ", node, ", result is: ",temp_adopted_nodes.keys())
                    temp_remove_nodes[node] = [time]

                if node not in cur_friend_pref:
                    cur_friend_pref[node] = {}
                    for nbr in graph.neighbors(node):
                        cur_friend_pref[node][nbr] = friend_pref[time - 1][node][nbr]

            friend_pref[time] = cur_friend_pref
            collective_adopted.append(temp_adopted_nodes)
            temp_adopted_nodes = new_adopted_nodes
            # print(time)
            total_past_reward[X_name][time] = cur_reward
            if not flag:
                # print("No adopt?, time is: ",time)
                last_time = time
                break

        friend_pref[0] = friend_pref[last_time]
        # print("All the adopted nodes:", len(temp_remove_nodes.keys()))
        pref_mse_list[X_name] = compute_pref_mse(friend_pref[0], graph)

    for X_name in total_past_reward.keys():
        process_result[X_name] = {}
        for (time,temp_reward) in total_past_reward[X_name].items():
            process_result[X_name][time] = 0
            for node in temp_reward.keys():
                process_result[X_name][time] += temp_reward[node][1]

    result_reward = {}
    for node in graph.nodes():
        temp_reward = []
        for past_reward in total_past_reward.values():
            for time in past_reward.keys():
                if len(past_reward[time]) == 0:
                    continue
                if node in past_reward[time] and len(past_reward[time][node][0]) > 0:
                    temp_reward.append(past_reward[time][node][1])
        # print(np.nansum(temp_reward))
        # print(temp_reward)
        result_reward[node] = np.nansum(temp_reward)
        # print("result reward for node: ",node," is ",result_reward[node])
    nx.set_node_attributes(graph, 'reward', result_reward)
    result_adopted = {}
    for single_adopted in collective_adopted:
        for node in single_adopted:
            if node in result_adopted:
                result_adopted[node] += 1
            else:
                result_adopted[node] = 1

    return graph, result_adopted, reputation


def plot_reward(Graph, individual_type, policy, X_list, friend_est_pref, adopted_node,
                time_step=3,eps_greedy=0.5, plot_state = False, process_result = None, pref_mse_list = None):
    # print("Total_time_step:", time_step)
    if process_result is None:
        process_result = {}
    if pref_mse_list is None:
        pref_mse_list = {}
    result__graph, result_adopted, reputation = graph_simulation(graph=Graph,
                                                     individual_type=individual_type,
                                                     friend_pref=friend_est_pref,
                                                     adopted_node=adopted_node,
                                                     X_list=X_list,
                                                     policy=policy,
                                                     total_time=time_step,
                                                     eps_greedy=eps_greedy,
                                                     process_result=process_result,
                                                     pref_mse_list=pref_mse_list)
    pos = nx.spring_layout(result__graph)
    node_color = []
    node_size = []
    node_reward = nx.get_node_attributes(result__graph, 'reward')
    total_reward = {}
    for (node, reward) in node_reward.items():
        if reward > 0:
            node_color.append('red')
        elif reward <= 0:
            node_color.append('blue')
        node_size.append(np.abs(reward) * 100)
        total_reward[node] = reward
    if plot_state:
        nx.draw(G=result__graph,
                pos=pos,
                cmap=plt.get_cmap('jet'),
                with_labels=True,
                node_size=node_size,
                labels={node: node for node in result__graph.nodes()},
                node_color=node_color)
        plt.title('reward ({}): {}'.format(individual_type, policy))
        plt.show()

    return total_reward, reputation


def plot_adoption(Graph, individual_type, policy, X_list, friend_est_pref,
                  adopted_node, time_step=3,eps_greedy=0.5, plot_state=False,process_result = None, pref_mse_list = None):
    print("Total_time_step:", time_step)
    if process_result is None:
        process_result = {}
    if pref_mse_list is None:
        pref_mse_list = {}
    result_graph, result_adopted, reputation = graph_simulation(graph=Graph,
                                                    individual_type=individual_type,
                                                    friend_pref=friend_est_pref,
                                                    adopted_node=adopted_node,
                                                    X_list=X_list,
                                                    policy=policy,
                                                    total_time=time_step,
                                                    eps_greedy=eps_greedy,
                                                    process_result=process_result,
                                                    pref_mse_list=pref_mse_list)
    pos = nx.spring_layout(result_graph)
    node_color = []
    node_size = []
    overall_adoption = []
    print("All adopted node's number: ", len(result_adopted))
    for node in result_graph.nodes():
        if node in result_adopted:
            node_color.append('red')
            node_size.append(result_adopted[node]*100)
            overall_adoption.append(result_adopted[node])
        else:
            print("Not adopted: ",node)
            node_size.append(300)
            node_color.append('blue')
            overall_adoption.append(0)

    if plot_state:
        nx.draw(G=result_graph,
                pos=pos,
                cmap=plt.get_cmap('jet'),
                with_labels=True,
                node_size=node_size,
                labels={node: node for node in Graph.nodes()},
                node_color=node_color)
        plt.title('adoption : {}'.format(policy))
        plt.show()
    return overall_adoption,reputation


def Graph_Initial_Check(Graph, X, adopted_node, homo_degree):
    initial_nodes = {'pos': 0, 'neg': 0}
    adopted_info = {}
    result_nodes = {}
    for n in Graph.nodes():
        temp_reward = logistic_func(X, Graph.node[n]['pref'])
        if temp_reward >0:
            initial_nodes['pos'] += 1
            result_nodes[n] = [0]
        else:
            initial_nodes['neg'] += 1
    for n in adopted_node:
        temp_reward = logistic_func(X, Graph.node[n]['pref'])
        if temp_reward > 0:
            adopted_info[n] = 'pos'
        else:
            adopted_info[n] = 'neg'

    # print(initial_nodes)
    # print(adopted_info)
    pos = nx.spring_layout(Graph)
    node_color = []
    node_size = []
    for node in Graph.nodes():
        if node in result_nodes:
            node_color.append('red')
            node_size.append(300)
        else:
            node_size.append(300)
            node_color.append('blue')

    nx.draw(G=Graph,
            pos=pos,
            cmap=plt.get_cmap('jet'),
            with_labels=True,
            node_size=node_size,
            labels={node: node for node in Graph.nodes()},
            node_color=node_color)
    plt.title('adoption : {}'.format(homo_degree))
