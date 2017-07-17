import numpy as np
import math
from history import *

class POMCP_Solver:

	def __init__(self, gamma, epsilon, timer, history, game, c, beta):
		"""
		creates a instance of a POMCP_Solver.
		:param gamma: this is the discount factor
		:param epsilon: this is the tolerance factor at which the rollout can be stopped
		:param timer: this is how long the search function is called
		:param history: this is the history from which the search function will be called
		:param game: the game that we pass in
		:param c: the constant which affects how much exploration vs. exploitation we want
		:param beta: the constant which measures human rationality
		"""
		self.gamma = gamma
		self.epsilon = epsilon
		self.timer = timer
		self.history = history
		self.game = game
		self.c = c
		self.beta = beta

	def search(self):
		"""
		The Search function as described in Silver et al.
		Returns the optimal action after the search tree has been constructed for timer timesteps.
		Also prunes the search tree so that it is now rooted at the optimal --robot-- action.
		
		NOTE: TO DO NEXT SEARCH, WE NEED TO RECEIVE A REAL OBSERVATION AND UPDATE THE ROOT AGAIN.
		ROOT MUST ALWAYS BE AT --HUMAN-- OR --NONE--.
		"""
		for _ in range(0, self.timer):
			#print(_)
			sample_state = self.history.sample_belief()
			self.simulate(sample_state, self.history, 0)
			# optimal_action = self.history.find_optimal_action(self.c)
			# print(optimal_action)

		optimal_action = self.history.find_optimal_action_non_aug()
		#print(self.history.children[optimal_action].visited)
		#print(self.history.children[(0,0,1)].visited)
		print(optimal_action)
		print(self.history.children[optimal_action].value)

		# print(self.history.visited)

		# t = 0
		# for robotaction in self.history.children:
		# 	t = t + self.history.children[robotaction].visited
		# print(t)

		i = 0
		x = self.history
		while len(x.children) > 0 and x != None:
			#print(i)
			# print(x)
			# print(x.children)
			x = x.children[(0, 0, 1)]
			i = i + 1
		
		self.history = self.history.children[optimal_action] #prune rest of search tree


		#THIS BUG WITH THE 16 LESS THING SHOULD BE FIXED

		# print(self.history.visited)
		# t = 0
		# for child in self.history.children:
		# 	t = t + sum(self.history.children[child].visited)
		# print(t)

		return optimal_action


	def rollout(self, state, depth, robot_action, human_action):
		"""
		This performs the rollout given an initial robot action. It selects randomly from a uniform
		distribution over robot and human actions.

		Returns the value of the rollout.
		"""

		#this and the statement below makes it such that this will only work for chef world - since not a sum
		if self.game.getReward(state) == 1:
			return 1

		if math.pow(self.gamma, depth) < self.epsilon:
			return 0

		next_state = self.game.getNextState(state, robot_action, human_action)
		
		random_index_robot = np.random.choice(range(0, len(self.game.getAllActions())))
		next_robot_action = self.game.getAllActions()[random_index_robot]

		random_index_human = np.random.choice(range(0, len(self.game.getAllObservations())))
		next_human_action = self.game.getAllObservations()[random_index_human]


		return self.gamma*self.rollout(next_state, depth + 1, next_robot_action, next_human_action)

	def simulate(self, state, history, depth):
		"""
		The Simulate function as described in Silver et al.
		Returns the reward received from moving through the search tree from an initial state and history.
		
		Most importantly, it constructs the search tree and updates the values of history nodes in
		the search tree.
		"""

		#adding this in, i think more correct:??
		if self.game.getReward(state) == 1:
			return 1

		if math.pow(self.gamma, depth) < self.epsilon:
			return 0

		if history.action_type == "human" or history.action_type == "None":
			#print(4)
			optimal_action = history.find_optimal_action(self.c)
			#print(self.history.children[optimal_action].value)
			
			if history.children[optimal_action].visited == 0:
				if history.action_type == "None":
					history.visited = history.visited + 1

				random_index_human = np.random.choice(range(0, len(self.game.getAllObservations())))
				human_action = self.game.getAllObservations()[random_index_human]
				value = self.gamma*self.rollout(state, depth, optimal_action, human_action)
				history.children[optimal_action].value = value
				history.children[optimal_action].increment_visited()

				#is this right? do we need the belief here? Also this means the tree has its end node
				#as a robot action.
				history.children[optimal_action].update_belief(state)

				history.children[optimal_action].create_children()
				history.children[optimal_action].children[human_action].increment_visited_human(state[1])
				history.children[optimal_action].children[human_action].update_human_value(value, state[1]) #update human's theta_list
				return value

			next_state, next_human_action = self.sample_boltzmann(state, history, optimal_action)
			next_history = history.children[optimal_action].children[next_human_action]
			if len(next_history.children) == 0:
				next_history.create_children()

			R = self.gamma*self.simulate(next_state, next_history, depth + 1)
			
			if history.action_type != "None":
				history.belief.append(state)

			#different for None and human
			history.increment_visited_human(state[1])
			if history.action_type != "None":
				history.update_human_value(R, state[1])
			
			history.children[optimal_action].increment_visited()
			history.children[optimal_action].update_value(R)

			#history.children[optimal_action].children[next_human_action].update_human_value(R, next_state[1]) #update human's theta_list
			
			return R

	def sample_boltzmann(self, state, history, robot_action):
		"""
		Samples a human action based on the future Q values.
		Returns the next state, future (sampled) human action, and reward.

		CHECK NP MATH HERE.
		"""
		list_of_probabilities = history.children[robot_action].compute_boltzmann_probabilities(self.beta, state[1])
		human_actions = self.game.getAllObservations()

		#print(list_of_probabilities)
		random_index_human = np.random.choice(range(0, len(human_actions)), p = list_of_probabilities)
		sample_human_action = human_actions[random_index_human]

		next_state = self.game.getNextState(state, robot_action, sample_human_action)
		next_reward = self.game.getReward(next_state)

		return next_state, sample_human_action






