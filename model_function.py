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


def friend_selection(Graph, individual_type, individual_node, policy, friend_pref, past_reward, X, time, parent_nodes,
                     recommended_nodes, ucb_counts,ucb_times,epsilon_greedy=0.5):
    new_reward = {}
    # if no neighbors
    availabel_nodes = [node for node in Graph.neighbors(individual_node)]
    availabel_nodes = list(set(availabel_nodes) - set(parent_nodes[individual_node]))
    availabel_nodes = list(set(availabel_nodes) - set(recommended_nodes[individual_node]))
    if len(availabel_nodes) == 0:
        # isolated node
        return -1, 0

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
    chosen_node = -1
    if policy == 'balanced':
        chosen_node = np.random.choice(availabel_nodes)
    elif policy == 'greedy':
        chosen_node = max(new_reward, key=new_reward.get)
    elif policy == 'e-greedy':
        if np.random.uniform(0, 1, 1) > epsilon_greedy:
            chosen_node = max(new_reward, key=new_reward.get)
        else:
            chosen_node = np.random.choice(availabel_nodes)
    elif policy == 'ucb':
        past_chosen = ucb_counts[individual_node]

        # past_chosen = Counter([past_reward[past_time][individual_node][0] for past_time in past_reward.keys()])
        for nbr in availabel_nodes:
            try:
                new_reward[nbr] += np.sqrt(2 * np.log1p(ucb_times) / (past_chosen[nbr] + 1))
            except Exception as e:
                print(e)
                print("wrong node, neighbor: ", individual_node, nbr)
                exit()
        chosen_node = max(new_reward, key=new_reward.get)
    elif policy == 'thompson':
        chosen_node = thompson_sample(Graph, individual_node, availabel_nodes, friend_pref[time - 1][individual_node], new_reward, X, 10)

    if individual_type == 'risk' or policy == 'ucb' or policy == 'thompson':
        new_reward[chosen_node] = estimate_reward(X, friend_pref[time - 1][individual_node][chosen_node])
    # print(friend_pref[time-1][individual_node][nbr]['mean'])
    # print(new_reward[chosen_node])
    if new_reward[chosen_node] < 0:
        return -1, 0

    recommended_nodes[individual_node].append(chosen_node)
    ucb_counts[individual_node][chosen_node] += 1
    return chosen_node, new_reward[chosen_node]


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

    return max(new_reward, key=new_reward.get)

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

    seed_node_num = int(node_num * 0.1)
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
        b_combinations = b_combinations.dot(eigenvects[:, :2].T).T
        # print(b_combinations)
        # new_b_combinations = b_combinations_.dot(eigenvects[:,-3:-1].T).T
        # print(new_b_combinations)
    elif homophily_degree == 'medium':
        b_combinations = b_combinations.dot(eigenvects[:, 16:18].T).T
    elif homophily_degree == 'weak':
        b_combinations = b_combinations.dot(eigenvects[:, 42:44].T).T
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
            ucb_counts[n][nbr] = 0
    ucb_times = 0
    name_count = 1
    pref_mse_list['0_0_0'] = compute_pref_mse(friend_pref[0], graph)



    for X in X_list:
        temp_adopted_nodes = copy.deepcopy(adopted_node)
        parent_recommends = {}
        recommended_nodes = {}
        for n in graph.nodes():
            parent_recommends[n] = []
            recommended_nodes[n] = []
        last_time = total_time-1
        X_name = ' '.join(str(e) for e in X) + '_' + str(name_count)
        name_count += 1
        total_past_reward[X_name] = {}


        for time in range(1, total_time):
            # print('time is: ', time)
            flag = False
            cur_reward = {}
            cur_friend_pref = {}
            if len(temp_adopted_nodes) <= 0:
                print("No adopted nodes, error!")
                break
            for node in graph.nodes():

                if node in temp_adopted_nodes and len(graph.neighbors(node)) > 0:
                    # adopted recommendation stratedy
                    chosen_node, reward = friend_selection(Graph=graph,
                                                           individual_type=individual_type,
                                                           individual_node=node,
                                                           friend_pref=friend_pref,
                                                           X=X,
                                                           time=time,
                                                           policy=policy,
                                                           past_reward=total_past_reward[X_name],
                                                           parent_nodes=parent_recommends,
                                                           recommended_nodes=recommended_nodes,
                                                           ucb_counts=ucb_counts,
                                                           ucb_times=ucb_times,
                                                           epsilon_greedy=eps_greedy)


                    cur_friend_pref[node] = {}
                    ucb_times += 1
                    if chosen_node == -1:
                        # print("Node: ",node," choose no node!")
                        for nbr in graph[node]:
                            cur_friend_pref[node][nbr] = friend_pref[time - 1][node][nbr]
                        continue
                    friend_reward = logistic_func(X, graph.node[chosen_node]['pref'])
                    cur_reward[node] = [chosen_node, friend_reward]
                    # print("Time: ", time," Current reward is ", friend_reward)
                    if friend_reward >= 0:
                        # print('recommendation successful!')
                        node_feedback = 1
                        reputation[node] += 1
                        if chosen_node not in temp_adopted_nodes:
                            temp_adopted_nodes[chosen_node] = [time]

                        else:
                            temp_adopted_nodes[chosen_node].append(time)
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
                        if nbr != chosen_node:
                            cur_friend_pref[node][nbr] = friend_pref[time - 1][node][nbr]

                if node not in cur_friend_pref:
                    cur_friend_pref[node] = {}
                    for nbr in graph.neighbors(node):
                        cur_friend_pref[node][nbr] = friend_pref[time - 1][node][nbr]

            friend_pref[time] = cur_friend_pref
            total_past_reward[X_name][time] = cur_reward
            if not flag:
                # print("No adopt?, time is: ",time)
                last_time = time
                break
        collective_adopted.append(temp_adopted_nodes)

        friend_pref[0] = friend_pref[last_time]
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
                if node in past_reward[time] and past_reward[time][node][0] is not None:
                    temp_reward.append(past_reward[time][node][1])
        # print(np.nansum(temp_reward))
        # print(temp_reward)
        result_reward[node] = np.nansum(temp_reward)
        # print("result reward for node: ",node," is ",result_reward[node])
    nx.set_node_attributes(graph, 'reward', result_reward)
    # result_adopted = {}
    # for single_adopted in collective_adopted:
    #     for node in single_adopted:
    #         if node in result_adopted:
    #             result_adopted[node].extend(single_adopted[node])
    #         else:
    #             result_adopted[node] = single_adopted[node]

    return graph, collective_adopted, reputation


def plot_reward(Graph, individual_type, policy, X_list, friend_est_pref, adopted_node,
                time_step=3,eps_greedy=0.5, plot_state = False, process_result = None, pref_mse_list = None):
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
    pos = nx.circular_layout(result__graph)
    node_color = []
    node_size = []
    node_reward = nx.get_node_attributes(result__graph, 'reward')
    total_reward = 0
    for (node, reward) in node_reward.items():
        if reward > 0:
            node_color.append('red')
        elif reward <= 0:
            node_color.append('blue')
        node_size.append(np.abs(reward) * 100)
        total_reward += reward
    if plot_state:
        nx.draw(G=result__graph,
                pos=pos,
                cmap=plt.get_cmap('jet'),
                with_labels=True,
                node_size=node_size,
                labels={node: node for node in result__graph.nodes()},
                node_color=node_color)
        plt.title('reward ({}): {}'.format(individual_type, policy))

    return total_reward, reputation


def plot_adoption(Graph, individual_type, policy, X_list, friend_est_pref,
                  adopted_node, time_step=3,eps_greedy=0.5, plot_state=False,process_result = None, pref_mse_list = None):
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
    pos = nx.circular_layout(result_graph)
    node_color = []
    node_size = []
    overall_adoption = []
    for single_adopted in result_adopted:
        temp_adoption = 0
        for node in result_graph.nodes():
            if node in single_adopted:
                node_color.append('red')
                node_size.append(len(single_adopted[node]) * 100)
                temp_adoption += len(single_adopted[node])
            else:
                node_size.append(200)
                node_color.append('blue')
        overall_adoption.append(temp_adoption)

    if plot_state:
        nx.draw(G=result_graph,
                pos=pos,
                cmap=plt.get_cmap('jet'),
                with_labels=True,
                node_size=node_size,
                labels={node: node for node in Graph.nodes()},
                node_color=node_color)
        plt.title('adoption : {}'.format(policy))

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
