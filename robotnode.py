class RobotNode:
	def __init__(self, game, value, visited):
		"""
		Initializes the RobotNode class, which is a node in the search tree storing the robot action.
		:param game: the game we are trying to solve.
		:param value: the value that was just received when the robot action was taken.
		:param visited: the number of times we have taken the action - should be 1.
		"""
		self.type = "robot"
		self.game = game
		self.observations = self.game.getAllObservations()
		self.children = self.make_children()
		self.value = value
		self.visited = visited

	def make_children(self):
		"""
		Makes the children (human actions) of the RobotNode.
		"""
		children = []
		for action in self.observations:
			children.append("empty")

		return children

	def augmented_value(self, c):
		"""
		Returns the augmented value (value + exploration) of taking this robot action.
		"""
		#test to make sure this division isnt just 0
		return self.value + c/self.visited

	def update_value(self, reward):
		"""
		Averages the new reward with the current value to update it.
		:param reward: the reward just received
		"""
		val = self.value
		val = val + ((reward - val)/self.visited)
		self.value = val

	def update_visited(self):
		"""
		Increments the number of times this robot action has been taken.
		"""
		count = self.visited
		count = count + 1
		self.visited = count