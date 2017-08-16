class Game:

	def __init__(self, M, N, initial_position, gamma = 0.95):
		self.M = M
		self.N = N
		self.current_position = initial_position
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
		Returns an array of all grids with different possible current locations.
		"""
		array = []
		for x in range(M):
			for y in range(N):
				grid = [[0 for j in range(N)] for i in range(M)]
				grid[x][y] = 1
				array.append(grid)
		return array

	def getAllActions(self):
		"""
		Returns an array of all possible robot actions.
		"""
		return [(0,1),(0,-1),(1,0),(-1,0), "shutdown", "sample"]

	def getAllObservations(self):
		"""
		Returns an array of all possible human actions.
		"""
		return [(0,1),(0,-1),(1,0),(-1,0), "shutdown", "sample"]

	def getNextState(self, current_position, robot_action, human_action):

		x, y = current_position

		if not isinstance(robot_action, str) and not isinstance(human_action, str):
			if x + robot_action[0] + human_action[0] < M and y + robot_action[1] + human_action[1] < N:
				x = x + robot_action[0] + human_action[0]
				y = y + robot_action[1] + human_action[1]
				return (x, y)

		if not isinstance(robot_action, str):
			if x + robot_action[0] < M and y + robot_action[1] < N:
				x = x + robot_action[0]
				y = y + robot_action[1]

		if not isinstance(human_action, str):
			if x + human_action[0] < M and y + human_action[1] < N:
				x = x + human_action[0]
				y = y + human_action[1]			

		
