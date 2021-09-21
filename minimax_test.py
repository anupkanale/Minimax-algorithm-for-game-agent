import numpy as np

class defNode():
    def __init__(self, index, value, children):
        self.index = index
        self.children = children
        self.value = value


def staticEval(node):
    return node.value


def minimax(currentNode, depth, maxPlayer, alpha, beta):

    # termination condition
    if depth == 0 or len(currentNode.children) == 0:
        return staticEval(currentNode)

    if maxPlayer:
        # gather evaluation from min player for different nodes and pick max of those
        globalMax = -np.Inf
        for childNode in currentNode.children:
            childNode.value = minimax(childNode, depth-1, False, alpha, beta)
            globalMax = max(globalMax, childNode.value)  # coz max player
            alpha = max(alpha, globalMax)
            if beta <= alpha:
                break
        return globalMax
    else:
        # gather evaluation from max player for different nodes and pick min of those
        globalMin = np.Inf
        for childNode in currentNode.children:
            childNode.value = minimax(childNode, depth - 1, True, alpha, beta)
            globalMin = min(globalMin, childNode.value)  # coz min player
            beta = max(beta, globalMin)
            if beta <= alpha:
                break
        return globalMin


if __name__ == "__main__":

    ## Test case-- Sebastian Lague video example from youtube.com
    node14 = defNode(15, 9, [])
    node13 = defNode(14, 0, [])
    node12 = defNode(13, -4, [])
    node11 = defNode(12, -6, [])
    node10 = defNode(11, 1, [])
    node9 = defNode(10, 5, [])
    node8 = defNode(9, 3, [])
    node7 = defNode(8, -1, [])

    node6 = defNode(7, 0, [node13, node14])
    node5 = defNode(6, 0, [node11, node12])
    node4 = defNode(5, 0, [node9, node10])
    node3 = defNode(4, 0, [node7, node8])
    node2 = defNode(3, 0, [node5, node6])
    node1 = defNode(2, 0, [node3, node4])
    node0 = defNode(1, 0, [node1, node2])

    alpha, beta = -np.Inf, np.Inf
    eval = minimax(node0, 3, True, alpha, beta)
    print(eval)
    # minimax(rootNode, 3, True)