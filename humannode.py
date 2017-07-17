class HumanNode:
	def __init__(self, game, reward, theta):
		"""
		Initializes the HumanNode class, which is a node in the search tree storing the human action.
		:param game: the game we are trying to solve.
		:param reward: the reward that was just received when the human action was taken.
		:param theta: the particular theta value of the state that was used when creating the node.
		"""
		self.type = "human"
		self.game = game
		self.actions = self.game.getAllActions()
		self.children = self.make_children()
		self.theta_list = self.game.getAllTheta()
		self.value_list = self.make_value_list(reward, theta)
		self.visited_list = self.make_visited_list(theta)

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
			if self.children[i] == "empty":
				values.append(c)
			else:
				values.append(self.children[i].augmented_value(c))

		return self.actions[values.index(max(values))]

	def make_value_list(self, reward, theta):
		"""
		Creates the list which is used to store the value of that search node for each theta.
		:param reward: the reward receive at the time this search node was created
		:param theta: the theta value from the state used when this search node was created
		"""
		#should this be the reward or gamma^depth*reward
		l = [0 for _ in range(len(self.theta_list))]
		l[self.theta_list[self.theta_list.index(theta)]] = reward
		return l

	def make_visited_list(self, theta):
		"""
		Creates the list which stores how many times a particular theta value visited the node.
		:param theta: the theta value of the state used when this search node was created
		"""
		l = [0 for _ in range(len(self.theta_list))]
		l[self.theta_list[self.theta_list.index(theta)]] = 1
		return l

	def update_value(self, reward, theta):
		"""
		Updates the value of the search node.
		:param reward: the reward used to update the value
		:param theta: the particular theta from the state that resulted in the reward
		"""
		theta_index = self.theta_list.index(theta)
		val = self.value_list[theta_index]
		val = val + ((reward - val)/self.visited_list[theta_index])
		self.value_list[theta] = val

	def update_visited(self, theta):
		"""
		Increments the visit count of the search node for the particular theta that caused the visit.
		:param theta: the particular theta from the state that resulted in the visit to the node.
		"""
		theta_index = self.theta_list.index(theta)
		count = self.visited_list[theta_index]
		count = count + 1
		self.visited_list[theta_index] = count
