# Module file for implementation of ID3 algorithm.

# You can add optional keyword parameters to anything, but the original
# interface must work with the original test file.
# You will of course remove the "pass".

import os, sys, math
import numpy
# You can add any other imports you need.
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix

class Node:
  def __init__(self, H):
    self.H = H
    self.label = None
    self.split_name = None
    self.attrib_value = None
    self.children = []

  def rec_common_label(self, d):
    if not self.label is None:
      if self.label in d:
        d[self.label] = d[self.label] + 1
      else:
        d[self.label] = 1
    for c in self.children:
      c.rec_common_label(d)

  def most_common_label(self):
    l_dict = dict()
    self.rec_common_label(l_dict)
    best_label = None
    max_C = 0
    for l in l_dict:
      if l_dict[l] > max_C:
        max_C = l_dict[l]
        best_label = l
    return best_label

  def __str__(self):
    s = "{ "
    if not self.attrib_value is None:
      s = s + "attrib_value = '" + str(self.attrib_value) + "' "
    if not self.label is None:
      s = s + "label = '" + str(self.label) + "' "
    if not self.split_name is None:
      s = s + "split_name = '" + str(self.split_name) + "' "
    for c in self.children:
      s = s + str(c)
    s = s + "}"
    return s

class DecisionTree:
    def __init__(self, load_from=None):
        # Fill in any initialization information you might need.
        #
        # If load_from isn't None, then it should be a file *object*,
        # not necessarily a name. (For example, it has been created with
        # open().)
        print("Initializing classifier.")
        if load_from is not None:
            print("Loading from file object.")
            self.root_node = self.load(load_from)
        else:
          self.root_node = None

    def most_common_value(self, y):
      d = dict()
      max_count = 0
      max_y = None
      for e in y:
        if not e in d:
          d[e] = 1
        else:
          d[e] = d[e] + 1
        if d[e] > max_count:
          max_count = d[e]
          max_y = e
      return max_y

    # Create a root node for the tree
    # If all examples are positive, return the single-node tree Root, with label = +.
    # If number of predicting attributes is empty, then Return the single node tree Root,
    # with label = most common value of the target attribute in the examples.
    # Otherwise Begin
    #    A = The Attribute that best classifies examples.
    #    Decision Tree attribute for Root = A
    #    For each possible value, v_i, of A
    #       Add a new tree branch below Root, corresponding to the test A = v_i.
    #       Let Examples(v_i) be the subset of examples that have the value v_i for A
    #       If Examples(v_i) is empty
    #          Then below this new branch add a leaf node with label = most common target value in the examples
    #       Else
    #          below this new branch add the subtree ID3(Examples(v_i), Target_Attribute, Attributes - {A})
    # End
    def ID3(self, X, y, attribute_names):
        # Create a root node for the tree
        root = Node(self.entropy(y))
        # If all examples are positive/negative, return the single-node tree Root, with label = +/-.
        y_set = set(y)
        if len(y_set) <= 1:
           root.label = y[0]
           return root
        # If number of predicting attributes is empty, then Return the single node tree Root,
        # with label = most common value of the target attribute in the examples.
        if len(attribute_names) == 0:
           root.label = self.most_common_value(y)
           return root
        # Otherwise Begin
        #    A = The Attribute that best classifies examples.
        #    Decision Tree attribute for Root = A
        # 1 split all attributes...
        # 2 find which split has the highest IG
        best_split_y = None
        best_split_names = None
        best_split_name = None
        max_IG = -1.0
        for name in attribute_names:
          split_y, split_names = self.split(X[name], y)
          H = self.entropy_after_split(y, split_y)
          IG = root.H - H
          if IG > max_IG:
            best_split_y = split_y
            best_split_names = split_names
            best_split_name = name
            max_IG = IG
        # 3 use the best as splitter
        root.split_name = best_split_name
        # split matrix X correctly here
        best_X = self.split_X(X, y, X[root.split_name])
        attribute_names.remove(root.split_name) # Attributes - {A}

        #    For each possible value, v_i, of A
        for i in range(len(best_split_y)):
          v_y = best_split_y[i]
          cur_X = best_X[i]
        #       Add a new tree branch below Root, corresponding to the test A = v_i.
        #       Let Examples(v_i) be the subset of examples that have the value v_i for A
        #       If Examples(v_i) is empty
        #          Then below this new branch add a leaf node with label = most common target value in the examples
        #       Else
        #          below this new branch add the subtree ID3(Examples(v_i), Target_Attribute, Attributes - {A})
          if v_y == None or len(v_y) == 0:
            #raise ValueError()
            child = Node(self.entropy(v_y))
            child.attrib_value = best_split_names[i]
            child.label = self.most_common_value(y)
            root.children.append(child)
          else:
            child = self.ID3(cur_X, v_y, attribute_names)
            child.attrib_value = best_split_names[i]
            root.children.append(child)
        # End
        return root

    def split_X(self, X, y, split_attr):
      split_set = set(split_attr)
      matrix_list = []
      for e in split_set:
        cur_matrix = None
        for i in range(len(y)):
          if e == split_attr[split_attr.index[i]]:
            row_i = X[i:i+1]
            if cur_matrix is None:
              cur_matrix = row_i
            else:
              cur_matrix = cur_matrix.append(row_i)
        matrix_list.append(cur_matrix)
      return matrix_list

    def split(self, split_attr, y):
      split_set = set(split_attr)
      y_list = []
      name_list = []
      for e in split_set:
        cur_y = []
        for i in range(len(y)):
          if e == split_attr[split_attr.index[i]]:
            cur_y.append(y[i])
        y_list.append(cur_y)
        name_list.append(e)
      return y_list, name_list

    def entropy(self, y):
       set_y = set(y)
       n = len(y)
       E = 0
       for s in set_y:
          c = 0
          for e in y:
             if s == e:
                c = c + 1
          if c != 0:
             E = E - c/n*math.log(c/n, 2)
       return E

    def entropy_after_split(self, y, splits):
       n = len(y)
       E_after = 0
       for s in splits:
          E_s = self.entropy(s)
          E_after = E_after + len(s)/n * E_s
       return E_after

    #      class: like                                in code: y
    # attributes: cheese, sauce, spicy, vegetables    in code: attrs
    #     values: mozza
    def train(self, X, y, attrs, prune=False):
        # Doesn't return anything but rather trains a model via ID3
        # and stores the model result in the instance.
        # X is the training data, y are the corresponding classes the
        # same way "fit" worked on SVC classifier in scikit-learn.
        # attrs represents the attribute names in columns order in X.
        #
        # Implementing pruning is a bonus question, to be tested by
        # setting prune=True.
        #
        # Another bonus question is continuously-valued data. If you try this
        # you will need to modify predict and test.
        names = []
        for i in range(len(X.columns)):
          names.append(X.columns[i])
        self.root_node = self.ID3(X, y, names)

    def pred_rec(self, node, X):
      if node.split_name is None:
        return node.label
      else:
        for c in node.children:
          X_attr_name = X[node.split_name][X.index[0]]
          if c.attrib_value == X_attr_name:
            return self.pred_rec(c, X)
        # if we reach here something which is not trained
        r = node.most_common_label()
        if r is None:
          raise ValueError()
        return r

    def predict(self, instance):
      # Returns the class of a given instance.
      # Raise a ValueError if the class is not trained.
      if self.root_node is None:
        raise ValueError()

      return self.pred_rec(self.root_node, instance)
      #return random.choice(['yes', 'no']) # return a crappy prediction

    def test(self, X, y, display=False):
      # initially create stats for each y
      predictions = []
      #print(X.shape)
      for i in range(X.shape[0]):
        predictions.append(self.predict(X[i:i+1]))

      #c = 0
      #print("prediction <--> answer : result")
      #for i in range(len(y)):
      #  if predictions[i] == y[y.index[i]]:
      #    v = 1
      #    c = c + 1
      #  else:
      #    v = 0
      #  print("%s <--> %s : %d" % (predictions[i], y[y.index[i]], v))
      #print(c/len(y))

      # Returns a dictionary containing test statistics:
      # accuracy, recall, precision, F1-measure, and a confusion matrix.
      # If display=True, print the information to the console.
      # Raise a ValueError if the class is not trained.
      #result = { 'precision': precision_score(y, predictions, average=None),
      #     'recall': recall_score(y, predictions, average=None),
      #     'accuracy': accuracy_score(y, predictions),
      #     'F1': f1_score(y, predictions, average=None),
      #     'confusion-matrix': confusion_matrix(y, predictions)}
      result = {  'precision': precision_score(y, predictions, average=None),
            'recall': recall_score(y, predictions, average=None),
            'accuracy': accuracy_score(y, predictions),
            'F1': f1_score(y, predictions, average=None),
            'confusion-matrix': confusion_matrix(y, predictions)}
      if display:
        print(result)
      return result

    def __str__(self):
      # Returns a readable string representation of the trained
      # decision tree or "ID3 untrained" if the model is not trained.
      if self.root_node is None:
        return "ID3 untrained"
      else:
        return str(self.root_node)

    def save_rec(self, node, output):
      output.write(str(node.H) + "\n")
      if node.label is None:
        output.write("\n")
      else:
        output.write(str(node.label) + "\n")
      if node.split_name is None:
        output.write("\n")
      else:
        output.write(str(node.split_name) + "\n")
      if node.attrib_value is None:
        output.write("\n")
      else:
        output.write(str(node.attrib_value) + "\n")
      output.write(str(len(node.children)) + "\n")
      for c in node.children:
        self.save_rec(c, output)

    def save(self, output):
      # 'output' is a file *object* (NOT necessarily a filename)
      # to which you will save the model in a manner that it can be
      # loaded into a new DecisionTree instance.
      if self.root_node is None:
        return
      self.save_rec(self.root_node, output)

    def parse_rec(self, node, content, line_nr):
      node.H = 0.0
      node.label = None
      node.split_name = None
      node.attrib_value = None
      n_children = 0

      h_str = content[line_nr + 0]
      if h_str != '':
        node.H = float(h_str)

      label_str = content[line_nr + 1]
      if label_str != '':
        node.label = label_str

      split_name_str = content[line_nr + 2]
      if split_name_str != '':
        node.split_name = split_name_str

      attrib_value_str = content[line_nr + 3]
      if attrib_value_str != '':
        node.attrib_value = attrib_value_str

      n_str = content[line_nr + 4]
      if n_str != '':
        n_children = int(n_str)

      line_nr = line_nr + 5
      for i in range(n_children):
        child = Node(0.0)
        line_nr = self.parse_rec(child, content, line_nr)
        node.children.append(child)
      return line_nr

    def load(self, input):
      content = input.readlines()
      for i in range(len(content)):
        content[i] = content[i].replace('\n', '')
      # if number of rows is not divisible by 5 it's incompatible with this format
      if len(content) % 5 != 0:
        return None
      node = Node(0.0) # 0.0 is a dummy value
      self.parse_rec(node, content, 0)
      return node
