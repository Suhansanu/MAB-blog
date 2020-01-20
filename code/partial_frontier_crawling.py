import sys
import numpy as np
import random
import pandas as pd
sys.path.insert(0, './dt_decision_tree')
import dt_decision_tree

def random_sample(dataset, choice, query_size=10, whole=False):
    samples = []
    if set(choice.values()) != set('*'):
        v_names, v_values = [], []
        for p_name, p_value in choice.iteritems():
            if p_value != '*':
                v_names.append(p_name)
                v_values.append(p_value)
            
        samples = dataset[np.all(dataset[:, v_names] == v_values, axis=1)]
    else:
        samples = dataset
    if whole:
        return samples
    return random.sample(samples, min(query_size, len(samples)))


class AttributeNode():
    def __init__(self, attr_idx, predicate, dataset_partitions, unq_samples, target_idx, epoch_queries, scoring_system):
        self.attr_idx       = attr_idx
        self.predicate      = predicate.copy()
        self.target_idx     = target_idx
        self.attr_vals      = {}
        self.best_attr_val  = -1
        self.best_score, self.best_success, self.best_failure, self.best_osize, self.best_choice = 0.0, 0.0, 0.0, 0.0, predicate.copy()
        self.epoch_queries  = epoch_queries
        self.scoring_system = scoring_system

        self.initialise_values(dataset_partitions, unq_samples)
    
    def compute_scores(self, best=False):
        if best:
            self.attr_vals[self.best_attr_val][0] = self.get_score(self.attr_vals[self.best_attr_val])
            self.best_score, self.best_success, self.best_failure, self.best_osize = self.attr_vals[self.best_attr_val]
        else:
            for attr_val, res_tuple in self.attr_vals.iteritems():
                res_tuple[0] = self.get_score(res_tuple)
                # if res_tuple[0] < 0:
                #     print 'here', res_tuple, self.attr_idx, self.predicate, self.attr_vals
            
            # getting best attr_val
            max_score, max_attrval = -1.*sys.float_info.max, -1
            for attr_val, tup in self.attr_vals.iteritems():    
                score = tup[0]
                if score > max_score:
                    max_score = score
                    max_attrval = attr_val

            # updating the best attr_val 
            self.best_score, self.best_success, self.best_failure, self.best_osize = self.attr_vals[max_attrval]        
            self.best_attr_val = max_attrval
            self.best_choice[self.attr_idx] = max_attrval
            
            
    def get_score(self, res_tuple):
        success_init, failure_init = 1, 1
        if self.scoring_system == 'quality':
            return 1. * res_tuple[1] / res_tuple[2]
        elif self.scoring_system == 'unseen_quality':
            return 1. * res_tuple[1] / res_tuple[2] * (res_tuple[3] - res_tuple[2]) / (res_tuple[3])
        elif self.scoring_system == 'expected_reward':
            bag_size         = res_tuple[3]
            if bag_size == 0:
                print self.attr_idx, self.predicate, res_tuple
            est_unique_items = bag_size* (1 - np.exp(-1. * self.epoch_queries / bag_size)) if bag_size >= 100 \
                                else bag_size* (1. - (1 - 1. / bag_size)**(self.epoch_queries))
            return 1. * res_tuple[1] / res_tuple[2] * (res_tuple[3] - res_tuple[2]) / (res_tuple[3]) * est_unique_items
        elif self.scoring_system == 'stochastic_quality':
            return np.random.beta(res_tuple[1]+success_init, res_tuple[2]-res_tuple[1]+failure_init)
        elif self.scoring_system == 'stochastic_unseen_quality':
            return 1. * np.random.beta(res_tuple[1]+success_init, res_tuple[2]-res_tuple[1]+failure_init) * (res_tuple[3] - res_tuple[2]) / (res_tuple[3])
        elif self.scoring_system == 'sotchastic_expected_reward':
            bag_size         = res_tuple[3]
            if bag_size == 0:
                print self.attr_idx, self.predicate, res_tuple
            est_unique_items = bag_size* (1 - np.exp(-1. * self.epoch_queries / bag_size)) if bag_size >= 100 \
                                else bag_size* (1. - (1 - 1. / bag_size)**(self.epoch_queries))
            return 1. * np.random.beta(res_tuple[1]+success_init, res_tuple[2]-res_tuple[1]+failure_init) * (res_tuple[3] - res_tuple[2]) / (res_tuple[3]) * est_unique_items
        
    
    def initialise_values(self, dataset_partitions, unq_samples):
        attr_vals = np.unique(unq_samples[:,self.attr_idx])
        tuples  = np.array(pd.DataFrame(unq_samples[:,[self.attr_idx, self.target_idx]], columns=[self.attr_idx, self.target_idx])\
                                       .groupby(self.attr_idx) \
                                       [self.target_idx].aggregate([np.sum, np.size]).reset_index())
        tmp_predicate = self.predicate.copy()
        
        # stroring the results
        for tup in tuples:
            tmp_predicate[self.attr_idx] = tup[0]
            self.attr_vals[tup[0]] = [0.0, tup[1], tup[2], dataset_partitions.original_size(tmp_predicate)]
        
        self.compute_scores(best=False)
        
    
    def is_generic(self, curr_choice, best=False):
        if best:
            for attr_idx in curr_choice:
                if self.best_choice[attr_idx] != '*' and curr_choice != '*' and self.best_choice[attr_idx] != curr_choice[attr_idx]:
                    return False
        else:
            for attr_idx in curr_choice:
                if self.predicate[attr_idx] != '*' and curr_choice != '*' and self.predicate[attr_idx] != curr_choice[attr_idx]:
                    return False        
        return True
    
    def update_new_results(self, q_result, dataset_partitions, best=False):
        if best:
            refined_result = random_sample(q_result, self.best_choice, whole=True)
            self.attr_vals[self.best_attr_val][1]+= np.sum(refined_result[:, self.target_idx])
            self.attr_vals[self.best_attr_val][2]+= len(refined_result)
            self.compute_scores(best=best)
        else:
            tuples  = np.array(pd.DataFrame(q_result[:,[self.attr_idx, self.target_idx]], columns=[self.attr_idx, self.target_idx])\
                                       .groupby(self.attr_idx) \
                                       [self.target_idx].aggregate([np.sum, np.size]).reset_index())            
            
            # # double check
            # for qr in q_result:
            #     checking = True
            #     for attr, attr_val in self.predicate.iteritems():
            #         if attr_val == '*':
            #             continue
            #         elif qr[attr] != attr_val:
            #             checking = False
            # if not checking:
            #     print 'there'
            #     print self.predicate
            #     print q_result

            # updating the results
            for tup in tuples:
                # existing attribute value
                if tup[0] in self.attr_vals:
                    self.attr_vals[tup[0]][1] += tup[1]
                    self.attr_vals[tup[0]][2] += tup[2]
                    if self.attr_vals[tup[0]][2] > self.attr_vals[tup[0]][3]:
                        print q_result
                        print self.attr_vals
                        print tuples
                        print self.attr_idx, tup[0], self.predicate
                        raise Exception('Not generic!')
                # new attribute value encountered
                else:
                    tmp_predicate                = self.predicate.copy()
                    tmp_predicate[self.attr_idx] = tup[0]
                    self.attr_vals[tup[0]] = [0.0, tup[1], tup[2], dataset_partitions.original_size(tmp_predicate)]

                    
            self.compute_scores(best=best)

class Frontier:
    def __init__(self, attr_idxs, attr_values, target_idx, dataset_partitions, epoch_queries, scoring_system, verbose=True):
        self.attr_idxs     = attr_idxs
        self.attr_values   = attr_values
        self.target_idx    = target_idx
        self.dataset_partitions = dataset_partitions
        self.frontier_list = []
        self.unq_samples   = []
        self.unq_ids       = set()
        self.verbose       = verbose
        self.epoch_queries = epoch_queries
        self.scoring_system= scoring_system
        
    def add_new_attribute_value_pair(self, choice):
        for attr_idx, attr_value in choice.iteritems():
            if attr_value == '*':
                new_attribute = AttributeNode(attr_idx, choice.copy(), self.dataset_partitions, \
                                              random_sample(self.unq_samples, choice, whole=True), \
                                              self.target_idx, self.epoch_queries, self.scoring_system)
                self.remove_old(choice, attr_idx)
                self.frontier_list.append(new_attribute)
        
    def remove_old(self, choice, attr_idx):
        for i in xrange(len(self.frontier_list) - 1, -1, -1):
            if self.frontier_list[i].attr_idx == attr_idx and self.frontier_list[i].predicate == choice:
                del self.frontier_list[i]
        
    def update(self, samples, query_result, curr_choice, best=False):
        self.unq_samples   = np.unique(np.vstack(samples), axis=0)
        q_result           = np.unique(np.vstack(query_result), axis=0)
        unseen_filter      = [q[0] not in self.unq_ids for q in q_result]
        q_result           = q_result[unseen_filter]
        self.unq_ids.update(q_result[:,0])
        
        # updating only the best (or whole) attrval results in frontier
        for frontier in self.frontier_list:
            if frontier.is_generic(curr_choice, best):
                frontier.update_new_results(q_result, self.dataset_partitions, best=best)

        # double checking the size mismatch
        # if len(self.unq_ids) != len(self.unq_samples):
        #     print  'size mismatch'
        #     print  set(self.unq_samples[:,0]) - set(self.unq_ids)

    def best_frontier(self):
        max_frontier_score, max_frontier = -1, None
        for frontier in self.frontier_list:
            if self.verbose:
                print frontier.predicate, '(', frontier.attr_idx, frontier.best_attr_val, frontier.best_score, ')',  
            if frontier.best_score > max_frontier_score:
                max_frontier_score = frontier.best_score
                max_frontier       = frontier
        if self.verbose:
            print ''
        return max_frontier.best_choice
        
    
## select the best attribute on the frontier 
def best_frontier_selection(frontier, samples, dataset, dataset_partitions, attr_idxs, n_attr_values, target_idx, choice, verbose, extra):
    
    # choosing the best frontier
    new_choice = frontier.best_frontier().copy()

    
    # expanding the best frontier with new samples
    frontier.add_new_attribute_value_pair(new_choice.copy())
        
    
    return new_choice.copy()

## decision tree crawling for offline datasets
def partial_frontier_crawling(dataset, attr_idxs, target_idx, verbose=True, n_querys = [100], query_size=10, epoch_querys=10, sims=100, scoring_system='quality', extra=['']):
    partial_tree_results   = [[] for _ in n_querys]
    attr_values            = {attr_idx: np.unique(dataset[:,attr_idx])       for attr_idx in attr_idxs}
    n_attr_values          = {attr_idx: len(np.unique(dataset[:,attr_idx]))  for attr_idx in attr_idxs}
    dataset_partitions     = dt_decision_tree.DatasetPartitions(dataset, attr_idxs=attr_idxs, target_idx=target_idx)
    for sim in xrange(sims):
        try:
            if verbose:
                print 'sim', sim
            samples, choice, frontier = [], {}, Frontier(attr_idxs, attr_values, target_idx, dataset_partitions, 1.*epoch_querys*query_size, scoring_system, verbose)
            tmp_ctr = 0
            for iteration in xrange(n_querys[-1]):
                if iteration < max(epoch_querys, 10):
                    choice = {attr_idx:'*' for attr_idx in attr_idxs}
                else:
                    if iteration% epoch_querys == 0:  # implement at intervals of epoch querys
                        frontier.add_new_attribute_value_pair(choice.copy())
                        choice = best_frontier_selection(frontier, samples, dataset, dataset_partitions,\
                                                         attr_idxs, n_attr_values, target_idx, choice.copy(), verbose, extra)
                
                query_result = random_sample(dataset, choice, query_size=query_size)
                samples.append(query_result)
                frontier.update(samples, query_result, choice, best=False)
                
                if verbose:
                    print 'choice', choice
                
                if iteration+1 in n_querys:
                    tmp_samples = np.vstack(samples)
                    df_samples = pd.DataFrame(tmp_samples).drop_duplicates()
                    query_idx = n_querys.index(iteration+1)
                    partial_tree_results[query_idx].append(df_samples[target_idx].sum())
        except Exception, e:
            pass
    
    if verbose:
        for query_idx in xrange(len(n_querys)):
            print str(attr_idxs) + '-> '+ str(np.mean(partial_tree_results[query_idx]))+ ' +/- '+ str(np.std(partial_tree_results[query_idx]))
    else:
        return [np.mean(partial_tree_result) for partial_tree_result in partial_tree_results], \
            [np.std(partial_tree_result) for partial_tree_result in partial_tree_results]
    
    