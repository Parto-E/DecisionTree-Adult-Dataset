import numpy as np
import TreeNode as tn

def entropy(freq):
    freq_0 = freq[np.array(freq).nonzero()[0]]
    prob_0 = freq_0/float(freq_0.sum())
    return -np.sum(prob_0*np.log2(prob_0))


class DecisionTree(object):
    def __init__(self, max_depth = 10):
        self.root = None
        self.max_depth = max_depth
        self.Ntrain = 0
        self.nodes = []
    
    def fit(self, data, target, x_val, y_val):
        self.Ntrain = data.count()[0]  
        self.data = data              
        self.attributes = list(data)  
        self.target = target         
        self.labels = target.unique()
        
        ids = range(self.Ntrain)
        self.root = tn.TreeNode(ids = ids, entropy = self._entropy(ids), depth = 0)  
        queue = [self.root]

        while queue: 
            node = queue.pop()
    
            if node.depth < self.max_depth:
                node.children = self._split(node)
                if not node.children: 
                    self._set_label(node)
                queue += node.children
                self.nodes += node.children
            else:
                self._set_label(node)       
        self.acc = self.evaluation(x_val,y_val,True)      
                
    def _entropy(self, ids):
        if len(ids) == 0: return 0
        ids = list(ids)
        freq = np.array(self.target[ids].value_counts())
        return entropy(freq)

    def _set_label(self, node):
        target_ids = list(node.ids)
        node.set_label(self.target[target_ids].mode()[0])

    
    def _split(self, node):
        ids = node.ids 
        best_gain = 0
        best_splits = []
        best_attribute = None
        order = None
        sub_data = self.data.iloc[ids, :]
        for i, att in enumerate(self.attributes):
            values = self.data.iloc[ids, i].unique().tolist()
            if len(values) == 1: continue
            splits = []
            for val in values: 
                sub_ids = sub_data.index[sub_data[att] == val].tolist()
                splits.append([sub_id for sub_id in sub_ids])
                H = 0
            for split in splits:
                H += len(split)*self._entropy(split)/len(ids)
            gain = node.entropy - H 
            if gain > best_gain:
                best_gain = gain 
                best_splits = splits
                best_attribute = att
                order = values
        node.set_properties(best_attribute, order)
        child_nodes = [tn.TreeNode(ids = split,
                     entropy = self._entropy(split), depth = node.depth + 1) for split in best_splits]
        return child_nodes

    def predict(self, new_data):

        npoints = new_data.count()[0]
        labels = [None]*npoints
        for n in range(npoints):
            x = new_data.iloc[n, :] 
            node = self.root
            while node.children: 
                if x[node.split_attribute] in node.order:
                    node = node.children[node.order.index(x[node.split_attribute])]
                else:           
                    node.label = self._set_label(node)
                    break
            labels[n] = node.label  
        return labels
    
    def evaluation(self,x_test,y_test,show_acc=False):
        z = self.predict(x_test)
        count = 0
        for i in range(len(y_test)):
            if y_test[i] == z[i]:
                count += 1
        if show_acc:
            print((count/len(x_test))*100)
        self.acc = count/len(x_test)*100
        return count/len(x_test)*100
    
    def _eval(self,x_test,y_test,show_acc = False):
        z = self.predict(x_test)
        count = 0
        for i in range(len(y_test)):
            if y_test[i] == z[i]:
                count += 1
        if show_acc:
            print((count/len(x_test))*100)
        return count/len(x_test)*100 
    
    def pruning(self,x2,y2): 
        max_depth = 0
        for n in self.nodes:
            if n.depth > max_depth:
                max_depth = n.depth
        prun_list = []
        for n in self.nodes:
            if n.children:
                prun_list.append(n)
        prun_list1 = []
        for dep in range(1,max_depth):
            for n in prun_list:
                if n.depth == dep: prun_list1.append(n)   
        before_pr = len(prun_list1)
        acc = 100
        last_acc = self.acc
        nodes = []
        val_acc = []
        while acc +.2 >= last_acc: 
            pr_node = prun_list1.pop()
            if pr_node.children:
                last_node = pr_node.children
                pr_node.children = []
                self._set_label(pr_node)
                last_acc = self.acc
                acc = self.evaluation(x2,y2,False)
                before_pr -= 1
                nodes.append(before_pr)
                print(before_pr)
                val_acc.append(self._eval(x2,y2,True))
        pr_node.children = last_node        
        self.acc = self.evaluation(x2,y2)
        
        return nodes,val_acc,max_depth