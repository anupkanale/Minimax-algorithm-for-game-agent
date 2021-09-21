import numpy as np
import time


class minimaxGo:
    def __init__(self, n, previousBoard, currentBoard, stoneColor, komi):
        self.n = n
        self.previousBoard = previousBoard
        self.currentBoard = currentBoard
        self.stoneColor = stoneColor
        self.komi = komi

    def get_indices(self, numMovesRem):
        return [2, 1, 3, 0, 4]

    def get_depth(self, numMovesRem):
        if numMovesRem==2:
            return 1
        elif numMovesRem==1:
            return 0
        else:
            return 2

    def get_bestMove(self, numMovesRem):
        """
        function to pick next move.

        :return: next position to play
        """

        bestMove = "PASS"
        bestScore = -np.inf
        isMaxPlayer = False  # opponent plays next turn
        alpha, beta = -np.Inf, np.Inf

        # currentBoardCopy = [row[:] for row in self.currentBoard]
        # print("Is move valid: ", self.check_move_validity(0, 4, currentBoardCopy, self.stoneColor))

        if numMovesRem>=23 and self.currentBoard[2][2]==0:
            return [2, 2]

        indices = self.get_indices(numMovesRem)
        for ii in indices:
            for jj in indices:
                currentBoardCopy = [row[:] for row in self.currentBoard]
                if self.check_move_validity(ii, jj, currentBoardCopy, self.stoneColor):
                    currentBoardCopy[ii][jj] = self.stoneColor

                    depth = self.get_depth(numMovesRem)
                    score = self.minimax(currentBoardCopy, depth, isMaxPlayer, alpha, beta)
                    # print(score, (ii, jj))
                    if score > bestScore:
                        bestScore = score
                        bestMove = [ii, jj]

        # print(bestMove, bestScore, numMovesRem)
        return bestMove

    def chase_opponent(self, board):
        for ii in range(self.n):
            for jj in range(self.n):
                if board[ii][jj]==3-self.stoneColor:
                    neighbors = self.find_neighbors(ii, jj)
                    for point in neighbors:

                        if self.check_move_validity(point[0], point[1], board, self.stoneColor):
                            return point

    def minimax(self, board, depth, isMaxPlayer, alpha, beta):
        """
        Evaluate best possible score for player

        :param board:
        :param depth:
        :param isMaxPlayer:
        :param alpha:
        :param beta:
        :return: best possible score for player
        """

        # termination condition 1/2
        if depth==0:
            return self.get_staticEval(board)

        # find children of given board
        children = []
        if isMaxPlayer:
            sColor = self.stoneColor
        else:
            sColor = 3 - self.stoneColor

        indices = self.get_indices(numMovesRem)
        for ii in indices:
            for jj in indices:
                child = [row[:] for row in board]
                if self.check_move_validity(ii, jj, child, sColor):
                    child[ii][jj] = sColor
                    children.append(child)

        # termination condition 2/2
        if len(children) == 0:
            return self.get_staticEval(board)

        if isMaxPlayer:
            globalMax = -np.Inf
            for child in children:
                childScore = self.minimax(child, depth-1, False, alpha, beta)
                globalMax = max(childScore, globalMax)
                # alpha beta pruning
                alpha = max(alpha, globalMax)
                if beta <= alpha:
                    break
            return globalMax

        else:
            globalMin = np.Inf
            for child in children:
                childScore = self.minimax(child, depth-1, True, alpha, beta)
                globalMin = min(childScore, globalMin)
                # alpha beta pruning
                beta = max(beta, globalMin)
                if beta <= alpha:
                    break
            return globalMin

    def get_staticEval(self, board):
        """
        returns score of current board as a difference of xScore and oScore

        :param board:
        :return: score difference
        """
        # remove dead stones of opponent black stones
        numDead, newBoard = self.remove_dead_stones(board, 3-self.stoneColor)

        xCount = sum(row.count(1) for row in newBoard)
        oCount = sum(row.count(2) for row in newBoard) + self.komi

        if self.stoneColor==1:
            return xCount - oCount
        else:
            return oCount - xCount

    def check_move_validity(self, ii, jj, board, sColor):
        """
        Check validity of the move board[i][j]==stoneColor

        :param ii
        :param jj
        :param board
        :param sColor
        :return: T/F
        """
        # 1. check if position (ii, jj) is empty
        if board[ii][jj] != 0:
            return False
        board[ii][jj] = sColor

        #2. check if position (ii, jj) has liberty
        isOk = self.check_liberty(ii, jj, board)
        # print(isOk)
        if isOk:
            return True

        # 3. In case this move captures opponent stone without violating KO, remove
        # dead pieces and check again for liberty.
        removedStones, newBoard = self.remove_dead_stones(board, 3 - sColor)
        if not self.check_liberty(ii, jj, newBoard):
            return False
        else:
            if len(removedStones)!=0 and self.previousBoard == newBoard:
                return False
        # print("here")

        return True

    def check_liberty(self, ii, jj, board):
        """"
        Every stone remaining on the board must have at least one open point directly orthogonally adjacent
        (up, down, left, or right), or must be part of a connected group that has at least
        one such open point next to it

        :param ii
        :param jj
        :param board
        """
        # visualize_board(5, board, "other")
        chain = self.find_connected_chain(ii, jj, board)
        # print("chain", chain)
        for stone in chain:
            stoneNeighbors = self.find_neighbors(stone[0], stone[1])
            # print("stone neigh" ,stoneNeighbors)
            for point in stoneNeighbors:
                if board[point[0]][point[1]] == 0:
                    # print("liberty point", point)
                    return True
        return False

    def find_connected_chain(self, ii, jj, board):
        """
        Find connected chain of given color stones using DFS

        :param ii:
        :param jj:
        :param board:
        :return: connected_chain
        """
        dfsStack = [(ii, jj)]
        connectedChain = []
        while len(dfsStack)!=0:
            point = dfsStack.pop()
            connectedChain.append(point)
            neighborAllies = self.find_closest_allies(point, board)
            for stone in neighborAllies:
                if stone not in dfsStack and stone not in connectedChain:
                    dfsStack.append(stone)
        return connectedChain

    def find_closest_allies(self, point, board):
        """
        Find the closest neighbors of a stone, that are of same color.

        :param point:
        :param board:
        :return: list of allies
        """
        sColor = board[point[0]][point[1]]
        closestAllyList = []
        neighborList = self.find_neighbors(point[0], point[1])
        for stoneLoc in neighborList:
            if board[stoneLoc[0]][stoneLoc[1]] == sColor:
                closestAllyList.append(stoneLoc)

        return closestAllyList

    def find_neighbors(self, ii, jj):
        """
        find closest neighbors of given point
        :param ii:
        :param jj:
        :return:
        """
        neighborList = []
        if ii > 0:
            neighborList.append([ii - 1, jj])
        if ii < self.n-1:
            neighborList.append([ii + 1, jj])
        if jj > 0:
            neighborList.append([ii, jj - 1])
        if jj < self.n-1:
            neighborList.append([ii, jj + 1])
        return neighborList

    def remove_dead_stones(self, board, sColor):
        """
        Clear out opponent's dead stones from the board
        :param board
        :param sColor
        """
        newBoard = [row[:] for row in board]
        deadStones = []
        for ii in range(self.n):
            for jj in range(self.n):
                if board[ii][jj] == sColor:
                    if not self.check_liberty(ii, jj, board):
                        deadStones.append((ii, jj))
                        newBoard[ii][jj] = 0

        return deadStones, newBoard

    def check_ko_violation(self, board):
        """
        Check for KO rule violation
        :param board:
        :return: T if no violation
        """
        return True

def write_output(move):
    fHandle = open("output.txt", 'w')
    if move == "PASS":
        fHandle.write(move)
    else:
        fHandle.write(str(move[0]) + "," + str(move[1]))
    fHandle.close()


def visualize_board(n, board, fileName):
    """
    Visualize given board in a text file

    :param n:
    :param board:
    :param fileName:
    :return: --
    """
    if fileName=="input":
        fHandle = open("board_viz.txt", 'w')
    elif fileName=="output":
        fHandle = open("board_viz.txt", 'a')
    else:
        fHandle = open("board_viz_other.txt", 'a')

    for ii in range(n):
        for jj in range(n):
            if board[ii][jj] == 0:
                fHandle.write("_ ")
            elif board[ii][jj] == 1:
                fHandle.write("x ")
            elif board[ii][jj] == 2:
                fHandle.write("o ")
        fHandle.write("\n")
    fHandle.write("\n")

    fHandle.close()


def get_num_moves(board):
    if np.count_nonzero(np.array(board))<2:
        fHandle = open("num_moves.txt", 'w')
        numMovesRem = 24 - np.count_nonzero(np.array(board))
        fHandle.write(str(numMovesRem))
        fHandle.close()
        return numMovesRem
    else:
        fHandle = open("num_moves.txt", 'r')
        numMovesRem = int(fHandle.read().strip())
        fHandle.close()

        fHandle = open("num_moves.txt", 'w')
        numMovesRem -= 2
        fHandle.write(str(numMovesRem))
        fHandle.close()

        return numMovesRem


def read_input():
    """
    Read input file

    :return: stoneColor, currentBoard
    """
    fHandle = open("input.txt")
    fileLines = fHandle.readlines()
    stoneColor = int(fileLines[0])

    prevBoard = [[int(col) for col in row.rstrip('\n')] for row in fileLines[1:6]]
    currBoard = [[int(col) for col in row.rstrip('\n')] for row in fileLines[6:11]]
    fHandle.close()
    return stoneColor, currBoard, prevBoard


if __name__ == "__main__":
    tStart = time.time()
    N = 5
    stoneColor, currentBoard, prevBoard = read_input()
    komi = N/2
    visualize_board(N, currentBoard, "input")

    numMovesRem = get_num_moves(currentBoard)

    myAgent = minimaxGo(N, prevBoard, currentBoard, stoneColor, komi)
    move = myAgent.get_bestMove(numMovesRem)

    if move!="PASS":
        currentBoard[move[0]][move[1]] = stoneColor

    visualize_board(N, currentBoard, "output")
    write_output(move)

    # print(numMovesRem)
    # print("Best move: ", move)
    # print(time.time() - tStart)