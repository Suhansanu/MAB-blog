import sys
sys.path.insert(0, './dt_decision_tree')
import dt_decision_tree
import numpy as np
import random
import pandas as pd
from sklearn import tree
import operator



def random_sample(dataset, attr_idxs, attr_vals, query_size=10, whole=False):
    samples = []
    if set(attr_vals) != set('*'):
        v_names, v_values = [], []
        for p_name, p_value in zip(attr_idxs, attr_vals):
            if p_value != '*':
                v_names.append(p_name)
                v_values.append(p_value)
            
        samples = dataset[np.all(dataset[:, v_names] == v_values, axis=1)]
    else:
        samples = dataset
    if whole:
        return samples
    return random.sample(samples, min(query_size, len(samples)))

## select the best attribute and the corresponding attribteu value 
def best_selection_without_backtrack(samples, dataset, dataset_partitions, dt, attr_idxs, n_attr_values, target_idx, choice, dt_method, verbose, extra):
    unq_samples   = np.unique(np.vstack(samples), axis=0)

    #stopping criterion
    if '*' not in choice:
        return choice
    
    finding_attrs = []
    reverse_idx   = {}
    for si, selection in enumerate(choice):
        if selection == '*':
            finding_attrs.append(attr_idxs[si])
        reverse_idx[attr_idxs[si]] = si
    fltr_samples = np.vstack(random_sample(unq_samples, attr_idxs, choice, query_size=len(unq_samples), whole=True))
    # dltr_samples = np.vstack(random_sample(dataset, attr_idxs, choice, query_size=len(dataset), whole=True))
    
    if dt_method == 'info_gain':
        ord_attr_idxs, ord_igs, ord_vals = dt_decision_tree.best_information_gain(fltr_samples, finding_attrs, target_idx, extra)
    elif dt_method == 'succ_rate':
        ord_attr_idxs, ord_rates, ord_vals = dt_decision_tree.best_success_rate(fltr_samples, finding_attrs, target_idx, extra)
    elif dt_method == 'gini_gain':
        ord_attr_idxs, ord_gis, ord_vals = dt_decision_tree.best_information_gain(fltr_samples, finding_attrs, target_idx, extra)
    corr_ord_vals = dt_decision_tree.correct_ordered_val(dataset_partitions, fltr_samples, ord_attr_idxs, ord_vals, choice[:], reverse_idx)

    best_attr_idx = ord_attr_idxs[0]
    best_attr_val = corr_ord_vals[0][0][0]    # first 0 for the best among finding_attrs, next for the best attr_val, next for the id in tuple.
    
    choice[reverse_idx[best_attr_idx]] = best_attr_val
    return choice
    
    
    
## select the best attribute and the corresponding attribteu value 
def best_selection_with_backtrack(samples, dataset, dataset_partitions, dt, attr_idxs, n_attr_values, target_idx, choice, dt_method, verbose, extra):
    unq_samples   = np.unique(np.vstack(samples), axis=0)

    #stopping criterion
    if '*' not in choice:
        child_choice  = choice[:]
        if dt.curr.indegree() == 0:
            print dt.summary()
        parent_choice = dt.curr.neighbors(mode='IN')[0]['attr_vals'][:]
#         print child_choice, parent_choice
        if dt_decision_tree.correct_rate(dataset, unq_samples, attr_idxs, parent_choice, target_idx) > dt_decision_tree.correct_rate(dataset, unq_samples, attr_idxs, child_choice, target_idx):  # all other ord_vals for that attribute have less than that attribute
            # print choice, 'better parent', 
            # backtrack two steps up
            dt.backtrack() # backtrack on attribute
            dt.backtrack() # backtrack on attribute value
            choice = dt.curr['attr_vals'][:]
            # if verbose:
            #     print 'backtracking to', choice
            return choice
        else:
            # print choice, 'better child'
            return child_choice

    finding_attrs = []
    reverse_idx   = {}
    for si, selection in enumerate(choice):
        if selection == '*':
            finding_attrs.append(attr_idxs[si])
        reverse_idx[attr_idxs[si]] = si
    fltr_samples = np.vstack(random_sample(unq_samples, attr_idxs, choice, query_size=len(unq_samples), whole=True))
    # dltr_samples = np.vstack(random_sample(dataset, attr_idxs, choice, query_size=len(dataset), whole=True))

#     if finding_attrs == []:
#         print dt.summary()

    if dt_method == 'info_gain':
        ord_attr_idxs, ord_igs, ord_vals = dt_decision_tree.best_information_gain(fltr_samples, finding_attrs, target_idx, extra)
    elif dt_method == 'succ_rate':
        ord_attr_idxs, ord_rates, ord_vals = dt_decision_tree.best_success_rate(fltr_samples, finding_attrs, target_idx, extra)
    elif dt_method == 'gini_gain':
        ord_attr_idxs, ord_gis, ord_vals = dt_decision_tree.best_information_gain(fltr_samples, finding_attrs, target_idx, extra)

    corr_ord_vals = dt_decision_tree.correct_ordered_val(dataset_partitions, fltr_samples, ord_attr_idxs, ord_vals, choice[:], reverse_idx)


    if dt_method == 'succ_rate': # update the attributes for the new rate
        ord_rates_new = [corr_ord_val[0][1] for corr_ord_val in corr_ord_vals]
        zipped    = zip(ord_attr_idxs, ord_rates_new, corr_ord_vals)
        zipped.sort(key=operator.itemgetter(1), reverse=True)
        ord_attr_idxs, ord_rates_new, corr_ord_vals = zip(*zipped)

    best_attr_idx = ord_attr_idxs[0]
    best_attr_val = corr_ord_vals[0][0][0]    # first 0 for the best among finding_attrs, next for the best attr_val, next for the id in tuple.

    parent_choice = choice[:]
    choice[reverse_idx[best_attr_idx]] = best_attr_val
    child_choice = choice[:]
    
    
    
    if dt_decision_tree.correct_rate(dataset, unq_samples, attr_idxs, parent_choice, target_idx) < best_attr_val:  # all other ord_vals for that attribute have less than that attribute
        choice = child_choice
        dt.add_attribute_node(attr_idx=best_attr_idx, attr_vals=parent_choice[:])
        dt.add_value_node(attr_idxs=attr_idxs, attr_vals=child_choice[:])
#         print 'better hmmm child', choice
    else:
        # backtrack two steps up
        dt.backtrack() # backtrack on attribute
        dt.backtrack() # backtrack on attribute value
        choice = dt.curr['attr_vals'][:]
        # print 'backtracted to ', choice
        
#         print parent_choice, 'better hmmm parent', choice
    
    return choice
    
    
## decision tree crawling
def decision_tree_crawling(dataset, attr_idxs, target_idx, verbose=True, n_querys = [100], query_size=10, epoch_querys=10, choice_method= 'default', dt_method='info_gain', backtrack=True, sims=100, extra=['']):
    greedy_dt_results      = [[] for _ in n_querys]
    attr_values            = [np.unique(dataset[:,attr_idx]) for attr_idx in attr_idxs]
    n_attr_values          = [len(np.unique(dataset[:,attr_idx])) for attr_idx in attr_idxs]
    dataset_partitions     = dt_decision_tree.DatasetPartitions(dataset, attr_idxs=attr_idxs, target_idx=target_idx)
    for sim in xrange(sims):
        if verbose:
            print sim
        dt = dt_decision_tree.DecisionTree(attr_idxs=attr_idxs, attr_vals=['*' for _ in attr_idxs])
        samples, choice = [], []
        tmp_ctr = 0
        for iteration in xrange(n_querys[-1]):
            if iteration < max(epoch_querys, 10):
                choice = ['*' for _ in attr_idxs]
            else:
                if iteration% epoch_querys == 0:  # implement at intervals of epoch querys
                    if choice_method == 'default':
                        if backtrack:
                            choice = best_selection_with_backtrack(samples, dataset, dataset_partitions, dt, attr_idxs, n_attr_values, target_idx, choice, dt_method, verbose, extra)
                        else:
                            choice = best_selection_without_backtrack(samples, dataset, dataset_partitions, dt, attr_idxs, n_attr_values, target_idx, choice, dt_method, verbose, extra)
                    elif choice_method == 'population':
                        choice = ['*' for _ in attr_idxs]
            query_result = random_sample(dataset, attr_idxs, choice, query_size=query_size)
            samples.append(query_result)
            
            if verbose:
                print choice
            
            if iteration+1 in n_querys:
                tmp_samples = np.vstack(samples)
                df_samples = pd.DataFrame(tmp_samples).drop_duplicates()
                query_idx = n_querys.index(iteration+1)
                greedy_dt_results[query_idx].append(df_samples[target_idx].sum())
            
    if verbose:
        for query_idx in xrange(len(n_querys)):
            print str(attr_idxs) + '-> '+ str(np.mean(greedy_dt_results[query_idx]))+ ' +/- '+ str(np.std(greedy_dt_results[query_idx]))
    else:
        return [np.mean(greedy_dt_result) for greedy_dt_result in greedy_dt_results], [np.std(greedy_dt_result) for greedy_dt_result in greedy_dt_results]














# printing conditions
def best_dt_branch(samples, attr_idxs, n_attr_values, target_idx):
    # decision tree
    unq_samples = np.unique(np.vstack(samples), axis=0)
    clf = tree.DecisionTreeClassifier()
    
    features = []
    for attr_idx, n_attr_value in zip(attr_idxs, n_attr_values):
        ft = unq_samples[:, attr_idx]
        ft_onehot = np.identity(n_attr_value)[ft]
        features.append(ft_onehot)
    
    clf = clf.fit(np.concatenate(features, axis=1), unq_samples[:, target_idx])
    n_nodes        = clf.tree_.node_count
    children_left  = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature        = clf.tree_.feature
    threshold      = clf.tree_.threshold
    impurity       = clf.tree_.impurity
    value          = clf.tree_.value
    estimator      = clf

    parent = np.zeros(shape=n_nodes, dtype=np.int64)
    parent_cond = np.zeros(shape=n_nodes, dtype=np.int64)
    for p, child in enumerate(children_left):
        if child == -1:
            continue
        parent[child] = p 
        parent_cond[child] = 1

    for p, child in enumerate(children_right):
        if child == -1:
            continue
        parent[child] = p 
        
    parent[0] = -1
    best_node = np.argmax([1. * value[i].flatten()[1] / np.sum(value[i]) for i in range(n_nodes)])    
    if best_node == 0:
#         print best_node, ':', 'all'
        return ['*' for _ in attr_idxs]
#     print best_node, ':', 
    prts = []
    prts_cond = []
    first_pass = True
    j = best_node
    while first_pass or parent[j] != -1:
        prts_cond.append(parent_cond[j])
        j = parent[j]
        prts.append(j)
        first_pass = False
    
    ft_bag, bag_init = [], []
    ctr  = 0
    for bag, n_attr_value in enumerate(n_attr_values):
        ft_bag.append([])
        bag_init.append(ctr)
        for i in range(n_attr_value):
            ft_bag[bag].append(ctr)
            ctr+=1
    
    
    for prt, prt_c in reversed(zip(prts, prts_cond)):
#         print prt, '(' + 'X[:,'+ str(feature[prt]) +']<='+str(threshold[prt]) + ' ' + str(prt_c==1) + ')', '-',
        if prt_c:
            [bag.remove(feature[prt]) for bag in ft_bag if feature[prt] in bag]
        else:
            for bagid in range(len(ft_bag)):
                if feature[prt] in ft_bag[bagid]:
                    ft_bag[bagid] =[feature[prt]] 
#     print best_node, ' ', 1. * value[best_node].flatten()[1] / np.sum(value[best_node])
    choosen_bag = [random.choice(bag) for bag in ft_bag]
    return [choosen-bag_init[bagid] for bagid, choosen in enumerate(choosen_bag)]
# if one of the attributes is not used at all then we should return '*'


## binary decision tree crawling
def binary_decision_tree_crawling(dataset, attr_idxs, target_idx, verbose=True, n_querys = [100], choice_method= 'default', sims=100):
    greedy_dt_results      = [[] for _ in n_querys]
    attr_values            = [np.unique(dataset[:,attr_idx]) for attr_idx in attr_idxs]
    n_attr_values          = [len(np.unique(dataset[:,attr_idx])) for attr_idx in attr_idxs]
    for _ in xrange(sims):
        samples, choice = [], []
        for iteration in xrange(n_querys[-1]):
            if iteration < 10:
                choice = ['*' for _ in attr_idxs]
            else:
                if choice_method == 'default':
                    choice = best_dt_branch(samples, attr_idxs, n_attr_values, target_idx)
                elif choice_method == 'population':
                    choice = ['*' for _ in attr_idxs]
                
            
            query_result = random_sample(dataset, attr_idxs, choice)
            if len(query_result):
                samples.append(query_result)
            
            if iteration+1 in n_querys:
                tmp_samples = np.vstack(samples)
                df_samples = pd.DataFrame(tmp_samples).drop_duplicates()
                query_idx = n_querys.index(iteration+1)
                greedy_dt_results[query_idx].append(df_samples[target_idx].sum())
    if verbose:
        for query_idx in xrange(len(n_querys)):
            print str(attr_idxs) + '-> '+ str(np.mean(greedy_dt_results[query_idx]))+ ' +/- '+ str(np.std(greedy_dt_results[query_idx]))
    else:
        return [np.mean(greedy_dt_result) for greedy_dt_result in greedy_dt_results], [np.std(greedy_dt_result) for greedy_dt_result in greedy_dt_results]























