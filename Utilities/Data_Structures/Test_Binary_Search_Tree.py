from Binary_Search_Tree import Binary_Search_Tree


def test_add_element():

    tree = Binary_Search_Tree()

    tree.add_element(10)
    tree.add_element(4)
    tree.add_element(12)
    tree.add_element(2)
    tree.add_element(15)
    tree.add_element(5)
    tree.add_element(8)

    #         10
    #     4        12
    # 2      5        15
    #          8

    assert tree.root_node.value == 10

    assert tree.root_node.left_child.value == 4
    assert tree.root_node.left_child.left_child.value == 2
    assert tree.root_node.left_child.right_child.value == 5
    assert tree.root_node.left_child.right_child.right_child.value == 8

    assert tree.root_node.right_child.value == 12
    assert tree.root_node.right_child.right_child.value == 15

    assert tree.root_node.right_child.parent.value == 10
    assert tree.root_node.right_child.right_child.parent.value == 12





