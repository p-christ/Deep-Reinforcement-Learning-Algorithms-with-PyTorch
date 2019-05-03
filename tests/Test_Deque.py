from utilities.data_structures.Deque import Deque
from utilities.data_structures.Node import Node


def test_Deque_initialisation():

    deque = Deque(2, 1)
    assert all(deque.deque == [Node(0, (None,)), Node(0, (None,))])

def test_Deque_adding_elements():

    deque = Deque(2, 1)
    deque.add_element_to_deque(3, 5)
    deque.add_element_to_deque(2, 4)

    assert all(deque.deque == [Node(3, 5), Node(2, 4)])

    deque.add_element_to_deque(1, 2)

    assert all(deque.deque == [Node(1, 2), Node(2, 4)])

    deque.add_element_to_deque(-100, 0)
    deque.add_element_to_deque(0, 0)

    assert all(deque.deque == [Node(0, 0), Node(-100, 0)])