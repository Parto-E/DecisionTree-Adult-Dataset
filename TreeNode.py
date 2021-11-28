class TreeNode(object):
    def __init__(self, ids = [], children = [], entropy = 0, depth = 0):
        self.ids = ids           
        self.entropy = entropy  
        self.depth = depth      
        self.split_attribute = None
        self.children = children 
        self.order = None     
        self.label = None  

    def __repr__(self):
        if self.split_attribute != None :return self.split_attribute   
        else: return 'leaf'
    
    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute
        self.order = order

    def set_label(self, label):
        self.label = label
