#-----------------------------------------------------------------------------------------------------------------------------------------------#

                                                        # LIBRARY IMPORT & DATA READING

#-----------------------------------------------------------------------------------------------------------------------------------------------#



import pandas as pd 
import numpy as np

data = pd.read_csv("ML-algorithms\datasetID3.csv")
data.rename(columns= {"rain" : "Outlook", "normal" : "Humidity", "strong" : "Wind", "yes" : "PlayTennis"},  inplace=True)

 

#-----------------------------------------------------------------------------------------------------------------------------------------------#

                                                        # FUNCTIONS

#-----------------------------------------------------------------------------------------------------------------------------------------------#


# Function to calculate the entropy of an attribute
def calculateEntropy(S):
    total = sum(S)
    entropy = 0
    
    for n in S:
        if n > 0:  # Avoid log2(0) which is undefined
            entropy += (-n / total) * np.log2(n / total)  # NEED TO DEFINE BIGGER LOG FOR CASES WITH MORE BYTES
            
    return entropy


# Function to count the different values of the attributes 
def orderData(df,  attributes =["Outlook","Humidity","Wind"]):
    
    result = {}

    for i in range(len(df)):
        bool = (data.iloc[i]["PlayTennis"] == "yes")
        
        for attribute in attributes:
            if attribute not in result:
                result[attribute] = {}
                result[attribute]["total"] = [0,0]
                
            value = data.iloc[i][attribute] 
            
            if value not in result[attribute]:
                result[attribute][value] = [0,0]

            if bool:
                result[attribute][value][0] += 1
                result[attribute]["total"][0] += 1
            if not bool:
                result[attribute][value][1] += 1
                result[attribute]["total"][1] += 1
                
    return result



# Function to calculate the gain of a certain attribute (A)
def calculateGain(S,A):  
    data = orderData(S)[A]
    
    # EXAMPLE CALCULATED IN CLASS IS CORRECT
    # data =  {'total' : [16,4], 'strong' : [7,3], 'weak' : [9,1]}
    

    total = data.pop("total")    
    totalEntropy = calculateEntropy(total)

    for value in data:
       totalEntropy -= (sum(data[value])/sum(total)) * calculateEntropy(data[value])


    return totalEntropy


# PRINT THE GAIN FOR WIND
# print(calculateGain(data,"Wind"))





#-----------------------------------------------------------------------------------------------------------------------------------------------#

                                                        # TREE IMPLEMENTATION

#-----------------------------------------------------------------------------------------------------------------------------------------------#



class TreeNode:
    def __init__(self, attribute=None, is_leaf=False, label=None):
        self.attribute = attribute  # The attribute this node splits on
        self.is_leaf = is_leaf  # Whether the node is a leaf
        self.label = label  # The class label if it's a leaf node
        self.children = {}  # Children of this node

    def add_child(self, value, node):
        self.children[value] = node

    def predict(self, sample):
        if self.is_leaf:
            return self.label
        attribute_value = sample[self.attribute]
        if attribute_value in self.children:
            return self.children[attribute_value].predict(sample)
        else:
            return None  # Handle unseen attribute values



def choose_best_attribute(data, attributes, target_attribute):
    gains = {}
    for attribute in attributes:
        gains[attribute] = calculateGain(data, attribute)
    return max(gains, key=gains.get)  # Attribute with highest gain


def build_tree(data, attributes, target_attribute):
    # Base case: If all examples have the same label, create a leaf node
    labels = data[target_attribute].unique()
    if len(labels) == 1:
        return TreeNode(is_leaf=True, label=labels[0])
    
    # Base case: If no attributes are left, create a leaf with the most common label
    if len(attributes) == 0:
        most_common_label = data[target_attribute].mode()[0]
        return TreeNode(is_leaf=True, label=most_common_label)

    # Choose the best attribute to split on
    best_attribute = choose_best_attribute(data, attributes, target_attribute)
    
    # Create a root node for the current attribute
    root = TreeNode(attribute=best_attribute)
    
    # Get unique values for the chosen attribute and split data accordingly
    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value]
        
        # If the subset is empty, assign the most common label of the current node
        if subset.empty:
            most_common_label = data[target_attribute].mode()[0]
            root.add_child(value, TreeNode(is_leaf=True, label=most_common_label))
        else:
            # Remove the current attribute and recurse for child nodes
            new_attributes = [attr for attr in attributes if attr != best_attribute]
            child_node = build_tree(subset, new_attributes, target_attribute)
            root.add_child(value, child_node)
    
    return root



def predict(tree, sample):
    return tree.predict(sample)


print(predict(build_tree(data,["Outlook","Humidity","Wind"], "PlayTennis"), {"Outlook": "rain", "Humidity": "high", "Wind" : "weak"}))

