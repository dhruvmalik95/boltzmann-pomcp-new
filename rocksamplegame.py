class Rocksample_Game:

    def __init__(self, M, N, initial_position, rock_vector, theta_set, gamma = 0.95):
        """
        Initializes an instance of the CIRL Rocksample game.
        
        :param M: the x distance of the grid.
        :param N: the y distance of the grid.
        :param human: the input human.
        :param robot: the input robot.
        :param initial_position: the starting position coordinate of the game.
        :param rock_vector: a vector containing a tuple for each rock - the tuple contains the position and type of the rock.
        :param theta_set: the set of theta_vectors over which the robot has uncertainty. A theta vector has an entry for the reward             corresponding to each type of rock.
        :param gamma: the discount factor.
        """
        
        self.M = M
        self.N = N
        self.current_position = initial_position
        self.rock_vector = rock_vector
        self.theta_set = theta_set
        self.gamma = gamma

        self.grid = self.createGrid(M, N, initial_position)

    def createGrid(self, M, N, initial_position):
        """
        :param M: x distance of grid
        :param N: y distance of grid
        :param initial_position: a pair of indices in the grid

        Returns an M by N two dimensional array of 0's (and a 1 representing the inital position).
        """
        grid = [[0 for y in range(N)] for x in range(M)]
        grid[initial_position[0]][initial_position[1]] = 1
        return grid

    def getAllWorldStates(self):
        """
        Returns an array of all possible current locations.
        """
        
        array = []
        for x in range(M):
            for y in range(N):
                array.append((x,y))
        return array
    
    def getAllTheta(self):
        """
        Returns all possible values of theta as a Python list.
        """
        return range(len(self.theta_set))
    
    def getAllStates(self):
        """
        Returns all possible states in the POMDP game as a Python list.
        """
        return list(itertools.product(self.getAllWorldStates(), self.getAllTheta()))

    def getAllActions(self):
        """
        Returns an array of all possible robot actions.
        """
        return [(0,1),(0,-1),(1,0),(-1,0), (0,0), "sample"]

    def getAllObservations(self):
        """
        Returns an array of all possible human actions.
        """
        return [(0,1),(0,-1),(1,0),(-1,0), "sample"]
    
    def getReward(self, state, robot_action, human_action):
        """
        :param state: the augmented state.
        
        Returns the reward of being in this augmented state. Implicitly assumes that the sample action was called.
        """
        if robot_action == "sample" or human_action == "sample":
            position = state[0]
            theta = state[1]
            for rock in self.rock_vector:
                if rock[0][0] == position[0] and rock[0][1] == position[1]:
                    return self.theta_set[theta][rock[1]]
            return -1
        return 0
    
    def getNextState(self, current_state, robot_action, human_action):
        x, y = current_state[0]
        theta = current_state[1]
        
        if robot_action == "sample" and human_action == "sample":
            return current_state

        if robot_action == "sample":
            if x + human_action[0] < self.M and x + human_action[0] >= 0 and y + human_action[1] < self.N and y + human_action[1] >= 0:
                x = x + human_action[0]
                y = y + human_action[1]
            return ((x, y), theta)

        if human_action == "sample":
            if x + robot_action[0] < self.M and x + robot_action[0] >= 0 and y + robot_action[1] < self.N and y + robot_action[1] >= 0:
                x = x + robot_action[0]
                y = y + robot_action[1]
            return ((x, y), theta)

        if x + robot_action[0] + human_action[0] < self.M and x + robot_action[0] + human_action[0] >= 0 and y + robot_action[1] + human_action[1] < self.N and y + robot_action[1] + human_action[1] >= 0:
            x = x + robot_action[0] + human_action[0]
            y = y + robot_action[1] + human_action[1]
        return ((x, y), theta)        

    def transition(self, initial_state, robot_plan, human_action, final_state):
        return