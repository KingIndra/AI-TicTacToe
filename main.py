import pygame, sys, random, copy, os, numpy as np, pickle, time, pprint

# CONSTANTS
WIDTH = 600
HEIGHT = 600
ROWS = 3
COLS = 3

SQSIZE = HEIGHT // COLS
LINE_WIDTH = 5

CIRCLE_RADIUS = 60
CIRCLE_WIDTH = 10
CROSS_WIDTH = 10
SPACE = 55

BG_COLOR = (0, 204, 204)
LINE_COLOR = (0,163,163)
CIRCLE_COLOR = (255,255,255)
CROSS_COLOR = (64, 64, 64)

# pygame setup
pygame.init()
screen = pygame.display.set_mode((600, 600))
pygame.display.set_caption('TIC TAC TOE <AI/>')
screen.fill(BG_COLOR)

def print_data(data):
    game_space_link = game_data['game_space_link']
    game_space = game_data['game_space']

    print('')
    print(f'number of games played: {game_data["no_of_games"]}')
    print('')
    print('GAME_SPACE_LINK: ')
    for state in game_space_link:
        print(f'{state} : {game_space_link[state]}') 
    print('')
    print('GAME_SPACE:')
    for state in game_space:
        print(f'{state} : {game_space[state]}')
    print('')

class Board:

    def __init__(self):
        self.squares = np.zeros((ROWS,COLS)).astype(int)
        self.empty_sqrs = self.squares
        self.marked_sqrs = 0
    
    def final_state(self, show=False):
        # return 0 if there is no win yet, return 1 if player 1 wins, return 2 if player 2 wins
        # vertical wins
        for col in range(COLS):
            if self.squares[0][col] == self.squares[1][col] == self.squares[2][col] != 0:
                # draw vertical winning line
                if show:
                    color = CIRCLE_COLOR if self.squares[0][col] == 2 else CROSS_COLOR
                    iPos = (col*SQSIZE+SQSIZE//2, 20)
                    fPos = (col*SQSIZE+SQSIZE//2, HEIGHT-20)
                    pygame.draw.line(screen, color, iPos, fPos, LINE_WIDTH)
                return self.squares[0][col]
        # horizontal wins
        for row in range(ROWS):
            if self.squares[row][0] == self.squares[row][1] == self.squares[row][2] != 0:
                # draw horizontal winnning line
                if show:
                    color = CIRCLE_COLOR if self.squares[row][0] == 2 else CROSS_COLOR
                    iPos = (20, row*SQSIZE+SQSIZE//2)
                    fPos = (WIDTH-20, row*SQSIZE+SQSIZE//2)
                    pygame.draw.line(screen, color, iPos, fPos, LINE_WIDTH)
                return self.squares[row][0]
        # desc diagonal win
        if self.squares[0][0] == self.squares[1][1] == self.squares[2][2] != 0:
            # draw desc winning line
            if show:
                color = CIRCLE_COLOR if self.squares[1][1] == 2 else CROSS_COLOR
                iPos = (20, 20)
                fPos = (WIDTH-20, HEIGHT-20)
                pygame.draw.line(screen, color, iPos, fPos, LINE_WIDTH)
            return self.squares[1][1]
        # asc diagonal win
        if self.squares[2][0] == self.squares[1][1] == self.squares[0][2] != 0:
            # draw asc winning line
            if show:
                color = CIRCLE_COLOR if self.squares[1][1] == 2 else CROSS_COLOR
                iPos = (20, HEIGHT-20)
                fPos = (WIDTH-20, 20)
                pygame.draw.line(screen, color, iPos, fPos, LINE_WIDTH)
            return self.squares[1][1]
        return 0

    def mark_sqr(self, row, col, player):
        self.squares[row][col] = player
        self.marked_sqrs += 1
    
    def empty_sqr(self, row, col):
        return self.squares[row][col] == 0
    
    def get_empty_sqrs(self):
        empty_sqrs = []
        for row in range(ROWS):
            for col in range(COLS):
                if self.empty_sqr(row, col):
                    empty_sqrs.append((row, col))
        return empty_sqrs

    def isfull(self):
        return self.marked_sqrs == 9
    
    def isempty(self):
        return self.marked_sqrs == 0


class BOT:

    def __init__(self):
        # constructor body
        self.game_stack = [] # initializing a stack to upadte game_space
        self.filename = 'game_data.pkl'
        # loading game_data pickel file
        if os.path.exists(self.filename):
            with open(self.filename, 'rb') as f:
                self.game_data = pickle.load(f)
        else:
            # initializing game_data dictionary if game_data file does not exist
            self.game_data = {'no_of_games':0, 'game_space_link':{}, 'game_space':{}}
        # initializing pointers for game_space_link and game_space dict
        self.game_space_link = self.game_data['game_space_link']
        self.game_space = self.game_data['game_space']
    
    # def clear(self):
    #     self.game_data

    def getHash(self, board_squares): # return hash of the given 3x3 matrix
        # repshaping 3x3 matrix to a 1d array
        arr = board_squares.reshape(9)
        # converting elements of array from int to str
        result = list(map(lambda x:str(x), arr))
        # join the result array to get our hash
        hash = ''.join([str(elem) for elem in result])
        # return hash
        return hash
    
    def getSquares(self, hash): # return 3x3 matrix of the given hash
        # initialize a 3x3 matrix with all zero elements
        arr = np.zeros((3,3)).astype(int)
        k = 0
        # traverse through the matrxi and update the values using the hash
        for i in range(3):
            for j in range(3):
                arr[i][j] = int(hash[k])
                k += 1
        # return 3x3 matrix
        return arr

    def chose_index(self, next_moves_dict, boardHash, s): # choosing the best possible using a probalbilty distrubution function on the q values of moves

        lst_pvals = []
        lst_moves = []

        # traversing through next_moves and appending the q values in lst_pvals
        for move in next_moves_dict:
            # random choice function give 0 probabilty below -708 that's why a lower limit is to to prevent all zeros in pval
            if next_moves_dict[move]['qval'] < -708:
                pval = -708
            else:
                pval = next_moves_dict[move]['qval']
            lst_pvals.append(pval)
            lst_moves.append(move)

        # converting lst_pvals into a numpy array to apply sigmoid function to it
        x = np.array(lst_pvals)*100
        weights = 1/(1 + np.exp(-x))

        # choosing a move according to it's probality abtained by q value
        try:
            choice = random.choices(lst_moves, weights=weights)[0]
        except ValueError as err:
            print(err)
            print(weights)

        # appending the current state to game_stack
        self.game_stack.append((boardHash, choice))

        # if s is an integer range from 0 to 6 rather than none it means we found our current state in game_space_link so this s is used to transform the move according to symmerty relation
        if s != None:
            choice = self.conter_point(choice, s)

        return choice

    def available_moves(self, board: Board): # returns a list of next available non symmetrical moves for the bot to play for a current game state
        # get all availabel empty squares by traversing the matrix
        empty_sqrs = board.get_empty_sqrs()
        # initialize a list to store all next possible state
        next_boards = []
        # initialize a list to store all possible next moves
        moves = []
        # initizlizing hash table
        hash_table = {}

        # filling the next board list
        for row, col in empty_sqrs:
            board_copy = copy.deepcopy(board)
            board_copy.mark_sqr(row, col, 2)
            next_boards.append(board_copy.squares)
            board_hash = self.getHash(board_copy.squares)
            hash_table[board_hash] = (row, col)

        j = 1
        # traversing the next_boards list and checking if there are any symmetrical states
        for board in next_boards:

            flag = True
            if j<len(next_boards):
                # one left rotation
                board_copy = np.rot90(board,k=1)
                for i in range(j, len(next_boards)):
                    if np.array_equal(board_copy, next_boards[i]):
                        flag = False
                # two left rotation
                board_copy = np.rot90(board,k=2)
                for i in range(j, len(next_boards)):
                    if np.array_equal(board_copy, next_boards[i]):
                        flag = False
                # three left rotation
                board_copy = np.rot90(board,k=3)
                for i in range(j, len(next_boards)):
                    if np.array_equal(board_copy, next_boards[i]):
                        flag = False
                # horizontal flip
                board_copy = np.flip(board,0)
                for i in range(j, len(next_boards)):
                    if np.array_equal(board_copy, next_boards[i]):
                        flag = False
                # vertical flip
                board_copy = np.flip(board,1)
                for i in range(j, len(next_boards)):
                    if np.array_equal(board_copy, next_boards[i]):
                        flag = False
                # flip across diagnol
                board_copy = self.switch(copy.deepcopy(board), [[(0,1),(1,0)],[(0,2),(2,0)],[(1,2),(2,1)]])
                for i in range(j, len(next_boards)):
                    if np.array_equal(board_copy, next_boards[i]):
                        flag = False
                # flip accros antidiagnol
                board_copy = self.switch(copy.deepcopy(board), [[(0,0),(2,2)],[(0,1),(1,2)],[(1,0),(2,1)]])
                for i in range(j, len(next_boards)):
                    if np.array_equal(board_copy, next_boards[i]):
                        flag = False
            
            # if the board has no symmetrical state present in next_boards then the move corresponding to it is stores in moves list using hash_table
            if flag:
                board_hash = self.getHash(board)
                move_ = hash_table[board_hash]
                moves.append(move_)
            else:
                pass
            j += 1

        return moves
    
    
    def switch(self, board, lst): 
        # switching the values of coordinate of a 3X3 matrix
        import copy
        board = copy.deepcopy(board)
        temp = 0
        for a,b in lst:
            temp = board[a[0]][a[1]]
            board[a[0]][a[1]] = board[b[0]][b[1]]
            board[b[0]][b[1]] = temp
        return board
    
    def conter_point(self, point, s): # returns the coordinate's s value equivalent point
        a,b = point
        # initializing a 3x3 matrix with all zeroes
        one = np.zeros((ROWS,COLS)).astype(int)
        # marking the given point as 1
        one[a][b] = 1
        # transforming the matrix according to given s value
        if s == 0:
            # one left rotation
            one = np.rot90(one, k=1)
        elif s == 1:
            # two left rotation
            one = np.rot90(one, k=2)
        elif s == 2:
            # three left rotation
            one = np.rot90(one, k=3)
        elif s == 3:
            # horizontal flip
            one = np.flip(one, 0)
        elif s == 4:
            # vertical flip
            one = np.flip(one, 1)
        elif s == 5:
            # flip across diagnol
            one = self.switch(one, [[(0,1),(1,0)],[(0,2),(2,0)],[(1,2),(2,1)]])
        elif s == 6:
            # flip accros antidiagnol
            one = self.switch(one, [[(0,0),(2,2)],[(0,1),(1,2)],[(1,0),(2,1)]])
        # traversing the matrix
        for i in range(3):
            for j in range(3):
                if one[i][j] == 1:
                    a, b = i, j
                    break
        # returning symmetrical equivalent point
        return a,b

    
    def symmetry(self, hash, boardHash): # returns if 2 states are symmetric and if they are then it also gives the s value to represent their realtion
        # getting the corresponding matrix of the first hash
        one = self.getSquares(hash)
        # making all possible symmetrical copies of the matrix obtained by the first hash argument
        copies = [
            np.rot90(one, k=1), 
            np.rot90(one, k=2), 
            np.rot90(one, k=3),
            np.flip(one, 0),
            np.flip(one, 1),
            self.switch(one, [[(0,1),(1,0)],[(0,2),(2,0)],[(1,2),(2,1)]]),
            self.switch(one, [[(0,0),(2,2)],[(0,1),(1,2)],[(1,0),(2,1)]])
        ]
        # initializing flag which is true is given 2 hashes are symmetrical
        flag = False
        # s represent the realtion between symmetrical states
        s = None
        # trverse the copies array and check if a copy's hash is equal to the second hash given
        for i, copy in enumerate(copies):
            copy_hash = self.getHash(copy)
            if copy_hash == boardHash:
                flag = True
                s = i
                break
        # return flag and s
        return flag, s

    def ai_chosen_square(self, board: Board):

        # check if 2 wins in next move
        for move in board.get_empty_sqrs():
            # make a copy for the current board state
            board_copy = copy.deepcopy(board)
            board_copy.mark_sqr(move[0], move[1], 2)
            # check the winning for player 2
            if board_copy.final_state() == 2:
                # return the winning move
                return move

        # check if 1 wins in next move
        for move in board.get_empty_sqrs():
            # make a copy for the current board state
            board_copy = copy.deepcopy(board)
            board_copy.mark_sqr(move[0], move[1], 1)
            # check the winning for player 1
            if board_copy.final_state() == 1:
                # return the winning move
                return move

        squares = board.squares
        # make a hash for the board squares
        boardHash = self.getHash(squares)
        # s value is a integer range from 0 to 6 which represents the symmetircal realtion of the current state, with its symmetrical state
        s = None

        # check if the current state is in game_space_link
        if boardHash in self.game_space_link.keys():
            # loading the symmetrical state hash and the s value to obtain the optimal next move from game_space
            boardHash_copy = boardHash
            boardHash = self.game_space_link[boardHash_copy]['hash']
            s = self.game_space_link[boardHash_copy]['s']

        # check if the current state not in game_space
        elif boardHash not in self.game_space.keys():

            flag = True

            # check if the current state is symmerical to a state present in game_space
            for hash in self.game_space.keys():
                if flag:
                    a, s = self.symmetry(hash, boardHash)
                    # if a state is symmertical to current state, then we store current stare in game_space_link with the s value
                    if a:
                        self.game_space_link[boardHash] = {'hash':hash, 's':s}
                        boardHash = hash
                        flag = False

            # if there is no symmertical state for the current state in game_space
            if flag:
                # store the current state as a hash key which will contain the available squares as its values with a q value for each available move
                self.game_space[boardHash] = {}
                for move in self.available_moves(board):
                    self.game_space[boardHash][move] = { 'qval': 1 } #'wins':0 , 'plays':0 , 

        # return the most optimal move.
        return self.chose_index(self.game_space[boardHash], boardHash, s)
    

    def update_game_state(self, final_state): # updating the q values of moves in game_data and saving the file as pickle
        
        # increamenting the number of games
        self.game_data['no_of_games'] += 1
        # updating the game_space_link dict of game_data dict
        # self.game_data['game_space_link'] = self.game_space_link

        # using the stack to traverse to all states of a game played and completed and updating q value of the moves according to win, lose or draw
        while len(self.game_stack)>0:
            hash, move = self.game_stack.pop()
            # if game was draw
            if final_state == 0:
                self.game_space[hash][move]['qval'] += 1*(1*2)
            # if player win
            elif final_state == 1:
                self.game_space[hash][move]['qval'] -= 1*(1*2)
            # if bot win
            else:
                self.game_space[hash][move]['qval'] += 1*(2*4)
        
        # updating the game_space dict of game_data dict
        # self.game_data['game_space'] = self.game_space

        # dumping the pickle file
        with open(self.filename, 'wb') as f:
            pickle.dump(self.game_data, f)


class MinMax:

    def __init__(self, player=2):
        self.player = player
        self.counter = 0
        self.cc = 1
    
    def getHash(self, board_squares): # return hash of the given 3x3 matrix
        # repshaping 3x3 matrix to a 1d array
        arr = board_squares.reshape(9)
        # converting elements of array from int to str
        result = list(map(lambda x:str(x), arr))
        # join the result array to get our hash
        hash = ''.join([str(elem) for elem in result])
        # return hash
        return hash

    def switch(self, board, lst): 
        # switching the values of coordinate of a 3X3 matrix
        board = copy.deepcopy(board)
        temp = 0
        for a,b in lst:
            temp = board[a[0]][a[1]]
            board[a[0]][a[1]] = board[b[0]][b[1]]
            board[b[0]][b[1]] = temp
        return board

    def available_moves(self, board: Board): # returns a list of next available non symmetrical moves for the bot to play for a current game state
        # get all availabel empty squares by traversing the matrix
        empty_sqrs = board.get_empty_sqrs()
        # initialize a list to store all next possible state
        next_boards = []
        # initialize a list to store all possible next moves
        moves = []
        # initizlizing hash table
        hash_table = {}

        # filling the next board list
        for row, col in empty_sqrs:
            board_copy = copy.deepcopy(board)
            board_copy.mark_sqr(row, col, 2)
            next_boards.append(board_copy.squares)
            board_hash = self.getHash(board_copy.squares)
            hash_table[board_hash] = (row, col)

        j = 1
        # traversing the next_boards list and checking if there are any symmetrical states
        for board in next_boards:

            flag = True
            if j<len(next_boards):
                # one left rotation
                board_copy = np.rot90(board,k=1)
                for i in range(j, len(next_boards)):
                    if np.array_equal(board_copy, next_boards[i]):
                        flag = False
                # two left rotation
                board_copy = np.rot90(board,k=2)
                for i in range(j, len(next_boards)):
                    if np.array_equal(board_copy, next_boards[i]):
                        flag = False
                # three left rotation
                board_copy = np.rot90(board,k=3)
                for i in range(j, len(next_boards)):
                    if np.array_equal(board_copy, next_boards[i]):
                        flag = False
                # horizontal flip
                board_copy = np.flip(board,0)
                for i in range(j, len(next_boards)):
                    if np.array_equal(board_copy, next_boards[i]):
                        flag = False
                # vertical flip
                board_copy = np.flip(board,1)
                for i in range(j, len(next_boards)):
                    if np.array_equal(board_copy, next_boards[i]):
                        flag = False
                # flip across diagnol
                board_copy = self.switch(copy.deepcopy(board), [[(0,1),(1,0)],[(0,2),(2,0)],[(1,2),(2,1)]])
                for i in range(j, len(next_boards)):
                    if np.array_equal(board_copy, next_boards[i]):
                        flag = False
                # flip accros antidiagnol
                board_copy = self.switch(copy.deepcopy(board), [[(0,0),(2,2)],[(0,1),(1,2)],[(1,0),(2,1)]])
                for i in range(j, len(next_boards)):
                    if np.array_equal(board_copy, next_boards[i]):
                        flag = False
            
            # if the board has no symmetrical state present in next_boards then the move corresponding to it is stores in moves list using hash_table
            if flag:
                board_hash = self.getHash(board)
                move_ = hash_table[board_hash]
                moves.append(move_)
            else:
                pass
            j += 1

        return moves
    
    def minimax(self, board: Board, maximizing):
        hash = self.getHash(board.squares)
        dp = self.hashMap.get(hash)
        if dp:
            return dp
        
        case = board.final_state()
        if case == 1:
            return 1, None # eval, move
        
        elif case == 2:
            return -1, None

        elif board.isfull():
            return 0, None

        if maximizing:
            max_eval = -2
            best_move = None

            if self.counter < self.cc:
                empty_sqrs = self.available_moves(board)
                self.counter += 1
            else:
                empty_sqrs = board.get_empty_sqrs()

            for (row, col) in empty_sqrs:
                temp_board = copy.deepcopy(board)
                temp_board.mark_sqr(row, col, 1)
                eval = self.minimax(temp_board, False)[0]
                if eval > max_eval:
                    max_eval = eval
                    best_move = (row, col)

            hash = self.getHash(board.squares)
            self.hashMap[hash] = max_eval, best_move

            return max_eval, best_move

        elif not maximizing: # minimizing
            min_eval = 2
            best_move = None

            if self.counter < self.cc:
                empty_sqrs = self.available_moves(board)
                self.counter += 1
            else:
                empty_sqrs = board.get_empty_sqrs()

            for (row, col) in empty_sqrs:
                temp_board = copy.deepcopy(board)
                temp_board.mark_sqr(row, col, self.player)
                eval = self.minimax(temp_board, True)[0]
                if eval < min_eval:
                    min_eval = eval
                    best_move = (row, col)

            hash = self.getHash(board.squares)
            self.hashMap[hash] = min_eval, best_move

            return min_eval, best_move

    def eval(self, main_board):
        start = time.time()
        self.hashMap = {}
        eval, move = self.minimax(main_board, False)
        end = time.time()
        print("The time of execution minimax function is :",(end-start) * 10**3, "ms")
        return move # row, col
    
    def reset(self):
        self.__init__()


class Game:

    def __init__(self):
        self.board = Board()
        # self.ai = AI()
        self.player = 1
        self.gamemode = 'ML'
        self.running = True
        self.show_lines()
    
    def make_move(self, row, col):
        self.board.mark_sqr(row, col, self.player)
        self.draw_fig(row, col)
        self.next_turn()

    def show_lines(self):
        # bg
        screen.fill(BG_COLOR)
        # vertical lines
        pygame.draw.line(screen, LINE_COLOR, (SQSIZE,0), (SQSIZE,HEIGHT), LINE_WIDTH)
        pygame.draw.line(screen, LINE_COLOR, (WIDTH-SQSIZE,0), (WIDTH-SQSIZE,HEIGHT), LINE_WIDTH)
        # horizontal lines
        pygame.draw.line(screen, LINE_COLOR, (0,SQSIZE), (WIDTH,SQSIZE), LINE_WIDTH)
        pygame.draw.line(screen, LINE_COLOR, (0,WIDTH-SQSIZE), (WIDTH,WIDTH-SQSIZE), LINE_WIDTH)

    def draw_fig(self, row, col):
        if self.player==1:
            # draw cross
            iPos1, fPos1 = (int(col*200+SPACE), int(row*200+200-SPACE)), (int(col*200+200-SPACE), int(row*200+SPACE))
            iPos2, fPos2 = (int(col*200+SPACE), int(row*200+SPACE)), (int(col*200+200-SPACE), int(row*200+200-SPACE))
            pygame.draw.line(screen, CROSS_COLOR, iPos1, fPos1, CROSS_WIDTH)
            pygame.draw.line(screen, CROSS_COLOR, iPos2, fPos2, CROSS_WIDTH)
        else:
            # draw circle
            center = (int(col*200+100), int(row*200+100))
            pygame.draw.circle(screen, CIRCLE_COLOR, (int(col*200+100), int(row*200+100)), CIRCLE_RADIUS, CIRCLE_WIDTH)
    
    def next_turn(self):
        self.player = self.player % 2 + 1
    
    def change_game_mode(self, mode):
        self.gamemode = mode
    
    def isover(self):
        return self.board.final_state(show=True) != 0 or self.board.isfull()

    def reset(self):
        self.__init__()

import asyncio

async def main():

    game = Game()
    ai = BOT()
    minmax = MinMax()

    while True:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                # g-gamemode
                if event.key == pygame.K_j:
                    game.reset()
                    minmax.reset()
                    game.change_game_mode("ML")

                if event.key == pygame.K_k:
                    game.reset()
                    minmax.reset()
                    game.change_game_mode("AI")

                if event.key == pygame.K_l:
                    game.reset()
                    minmax.reset()
                    game.change_game_mode("PVP")
                # r-restart
                if event.key == pygame.K_r:
                    game_mode = game.gamemode
                    if game_mode == "ML":
                        ai.update_game_state(game.board.final_state())
                        game.reset()
                        minmax.reset()
                        game.gamemode = game_mode
                    else:
                        print("you are not in ML mode, do not press R")

                if event.key == pygame.K_c:
                    game_mode = game.gamemode
                    game.reset()
                    minmax.reset()
                    game.gamemode = game_mode

                if event.key == pygame.K_d:
                    game_mode = game.gamemode
                    game.reset()
                    minmax.reset()
                    game.gamemode = game_mode
                    ai.game_data
                    
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                row = pos[1] // SQSIZE 
                col = pos[0] // SQSIZE
                if game.board.empty_sqr(row, col) and game.running:
                    game.make_move(row, col)
                    if game.isover():
                        game.running = False
        
 
        if game.gamemode == 'ML' and game.player == 2 and game.running:
            # ml methods
            row, col = ai.ai_chosen_square(game.board)
            game.make_move(row, col)
            if game.isover():
                game.running = False
        
        elif game.gamemode == 'AI' and game.player == 2 and game.running:
            # ai methods
            row, col = minmax.eval(game.board)
            game.make_move(row, col)
            if game.isover():
                game.running = False

        else: # if gamemode == "PVP"
            pass

        pygame.display.update()
        await asyncio.sleep(0)


asyncio.run(main())