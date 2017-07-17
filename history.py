import numpy as np
import math

class HistoryNode:

	def __init__(self, key, action_type, belief, game, value, visited):
		"""
		Generates an instance of the HistoryNode class.
		:param key: stores the current action/observation
		:param children: is a dictionary which maps actions to future HistoryNodes
		:param action_type: stores whether the key is a robot or human action
		:param belief: is a list that stores all states visited during the search process
		"""

		self.key = key
		self.action_type = action_type
		self.children = {}
		self.belief = belief
		self.game = game
		self.value = value
		self.visited = visited
		self.augmented_value = 0 #so i can put the c in the POMCP solver
		self.values_per_theta = self.create_values_per_theta()
		
		self.make_visited()
		
	def create_children(self):
		"""
		Creates the children for a particular history. 
		"""
		
		if self.action_type == "robot":
			for human_action in self.game.getAllObservations():
				new_node = HistoryNode(human_action, "human", [], self.game, 0, 0)
				self.children[human_action] = new_node
		elif self.action_type == "human" or self.action_type == "None":
			for robot_action in self.game.getAllActions():
				new_node = HistoryNode(robot_action, "robot", [], self.game, 0, 0)
				self.children[robot_action] = new_node
		else:
			print("LABELS ARE WRONG")

	def make_visited(self):
		if self.action_type == "robot" or self.action_type == "None":
			return

		theta_list = self.game.getAllTheta()
		visits_per_theta = []
		for theta in theta_list:
			visits_per_theta.append(0)

		self.visited = visits_per_theta

	def create_values_per_theta(self):
		"""
		Constructs the list for storing values of that taking that human action
		after prior history, for each theta value.
		"""
		if self.action_type == "robot":
			return []

		theta_list = self.game.getAllTheta()
		values_per_theta = []
		for theta in theta_list:
			values_per_theta.append(0)

		return values_per_theta

	def find_child(self, action):
		"""
		:param action: the action who's child history we are looking for

		Return the history who's key is the action we want. Returns None if history doesn't exist.
		"""
		return self.children[action]

	def find_optimal_action(self, c):
		"""
		:param c: the constant controlling exploitation vs. exploration

		Returns the optimal action of a history who's key is human or None, based on augmented values 
		of children.
		"""

		for k, v in self.children.items():
			if v.visited == 0:
				return k

		list_to_sort = []
		for robot_action in self.children:
			robot_action_value = self.compute_augmented_value(c, robot_action)
			list_to_sort.append((robot_action, robot_action_value))

		list_to_sort = sorted(list_to_sort, key=lambda x: x[1], reverse = True)
		return list_to_sort[0][0]

	def find_optimal_action_non_aug(self):
		"""
		Returns the optimal action of a history who's key is human or None, based on regular values 
		of children.
		"""

		list_to_sort = []

		for robot_action in self.game.getAllActions():
			robot_action_value = self.children[robot_action].value
			list_to_sort.append((robot_action, robot_action_value))

		list_to_sort = sorted(list_to_sort, key=lambda x: x[1], reverse = True)
		#print(list_to_sort)
		#print(" ")
		return list_to_sort[0][0]

	def sample_belief(self):
		"""
		Randomly samples and returns a state from the belief list of the history
		"""
		random_index = np.random.choice(range(0, len(self.belief)))
		return self.belief[random_index]

	def update_belief(self, state):
		"""
		Appends a state to the belief list of the history.
		"""
		self.belief.append(state)

	def increment_visited(self):
		"""
		Increments the number of times a history has been visited.
		"""
		self.visited = self.visited + 1

	def increment_visited_human(self, theta):
		if self.action_type == "None":
			self.visited = self.visited + 1

		if self.action_type == "human":
			theta_list = self.game.getAllTheta()
			theta_index = theta_list.index(theta)
			self.visited[theta_index] = self.visited[theta_index] + 1

	def update_value(self, reward):
		"""
		:param reward: Takes the current value and averages it with the reward just received.
		"""
		val = self.value
		val = val + ((reward - val)/self.visited)
		self.value = val

	def update_human_value(self, reward, theta):

		theta_list = self.game.getAllTheta()
		theta_index = theta_list.index(theta)

		val = self.values_per_theta[theta_index]
		number_times = self.visited[theta_index]
		
		if number_times == 0:
			self.values_per_theta[theta_index] = val + reward
			return
		
		val = val + ((reward - val)/number_times)
		self.values_per_theta[theta_index] = val

	def compute_augmented_value(self, c, robot_action):
		"""
		:param c: the constant controlling exploitation vs. exploration
		:param robot_action: the robot action who's value we want to determine.

		Returns the augmented value of taking a particular robot action.
		"""
		value = self.children[robot_action].value
		#print(self.action_type)
		if self.action_type == "human":
			#temporary shit. change this later on.
			v = max(self.visited) + 1
		else:
			v = self.visited
		augmented_value = value + c*math.pow((math.log(v) / self.children[robot_action].visited), 0.5)
		return augmented_value

	def compute_boltzmann_probabilities(self, beta, theta):
		"""
		:param beta: takes the beta parameter from the POMCP solver
		:param theta: takes the required theta value since each human action has a value for each theta

		Returns an np array of probabilities of getting each human action, ordered by human actions

		CHECK MY NP MATH HERE
		"""
		list_of_probabilities = []
		qValues = []

		theta_list = self.game.getAllTheta()
		theta_index = theta_list.index(theta)

		for human_action in self.game.getAllObservations():
			if self.children[human_action].visited[theta_index] == 0:
				qValues.append(0)
			else:
				qValues.append(self.children[human_action].values_per_theta[theta_index])
		
		print(qValues)
		qValues = np.array(qValues)
		expQ_values = np.exp(beta * qValues)
		total = sum(expQ_values)
		
		for x in list(expQ_values):
			list_of_probabilities.append(x / total)

		return np.array(list_of_probabilities)








