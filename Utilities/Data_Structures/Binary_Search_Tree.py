

class Node(object):

    def __init__(self, value, left_child=None, right_child=None, parent=None):
        self.value = value
        self.left_child = left_child
        self.right_child = right_child
        self.parent = parent

    def set_left_child(self, left_child):
        self.left_child = left_child

    def set_right_child(self, right_child):
        self.right_child = right_child

    def set_value(self, value):
        self.value = value


class Binary_Search_Tree(object):

    def __init__(self):
        self.root_node = None

    def add_element(self, value, starting_point=None):

        if starting_point is None:
            if self.root_node is None:
                self.root_node = Node(value)
            else:
                self.add_element(value, starting_point=self.root_node)

        else:
            if value <= starting_point.value:
                if starting_point.left_child is None:
                    starting_point.left_child = Node(value, parent=starting_point)
                else:
                    self.add_element(value, starting_point=starting_point.left_child)

            else:
                if starting_point.right_child is None:
                    starting_point.right_child = Node(value, parent=starting_point)
                else:
                    self.add_element(value, starting_point=starting_point.right_child)





