import numpy as np
import math
class Root:
	def __init__(self, game, belief, visited):
		"""
		Initializes the root of the search tree.
		:param game: the game that we want to solve.
		:param belief: the initial belief state that the robot has.
		:param visited: the number of times we have visited the root - should initialize to 1.
		"""
		self.type = "root"
		self.game = game
		self.actions = self.game.getAllActions()
		self.children = self.make_children()
		self.belief = belief
		self.visited = visited

	def make_children(self):
		"""
		Makes the children (robot actions) of the HumanNode.
		"""
		children = []
		for action in self.actions:
			children.append("empty")

		return children

	def optimal_action(self, c):
		"""
		Returns the optimal robot action to take from this search node.
		:param c: the constant that controls how much exploration should be done.
		"""
		# for i in range(0, len(self.children)):
		# 	if self.children[i] == "empty":
		# 		return self.actions[i]

		values = []
		for i in range(0, len(self.children)):
			#print(2)
			if self.children[i] == "empty":
				#print(3)
				values.append(c)
			else:
				#print(4)
				values.append(self.children[i].augmented_value(c))

		return self.actions[values.index(max(values))]

	def sample_belief(self):
		"""
		Randomly samples an initial state from the belief state.
		Returns the sampled state.
		"""
		random_index = np.random.choice(range(0, len(self.belief)))
		return self.belief[random_index]

	def update_visited(self, theta):
		"""
		Increments the number of times we have visited the root by 1.
		:param theta: this parameter is here simply to make the code work with HumanNodes.
		"""
		count = self.visited
		count = count + 1
		self.visited = count

	def update_value(self, reward, theta):
		"""
		We do not update the value of the root (can be easily computed with self.optimal_action(0)).
		"""
		return