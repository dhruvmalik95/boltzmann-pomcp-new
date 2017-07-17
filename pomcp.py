import numpy as np
import math

from humannode import *
from robotnode import *
from timeit import default_timer as timer
import requests
from timeout import *

class POMCP_Solver:
	def __init__(self, gamma, epsilon, timer, history, game, c, beta, behavior):
		"""
		creates an instance of a POMCP_Solver.
		:param gamma: this is the discount factor
		:param epsilon: this is the tolerance factor at which the rollout can be stopped
		:param timer: this is how long the search function is called
		:param history: this is the history from which the search function will be called
		:param game: the game that we pass in
		:param c: the constant which affects how much exploration vs. exploitation we want
		:param beta: the constant which measures human rationality
		:param behavior: the model of human rationality (rational or boltzmann)
		"""
		self.gamma = gamma
		self.epsilon = epsilon
		self.timer = timer
		self.history = history
		self.game = game
		self.c = c
		self.beta = beta
		self.behavior = behavior
		self.actions = self.game.getAllActions()
		self.observations = self.game.getAllObservations()
		self.theta_list = self.game.getAllTheta()
		self.data = []

	def search(self):
		"""
		The search function as described in Silver et al. Samples a start state for self.timer iterations.
		Prints optimal action and its value after iterations are complete.
		"""
		start_0 = timer()
		start = start_0
		for _ in range(0, self.timer):
			if _ % 100000 == 0:
				print(_)
			# if _ % 100000 == 0:
			# 	print(_)
			sample_state = self.history.sample_belief()
			self.simulate(sample_state, self.history, 0)

			# t = timer()
			# if t - start > 0.05 and _ > 4:
			# 	optimal_action = self.history.optimal_action(0)
			# 	optimal_child = self.history.children[self.actions.index(optimal_action)]
			# 	self.data.append((t-start_0, optimal_child.value))
			# 	start = t

			# if timer() - start_0 > 4*60:
			# 	return

			if _ > 4 and _%500==0:
				optimal_action = self.history.optimal_action(0)
				optimal_child = self.history.children[self.actions.index(optimal_action)]
				self.data.append((_,optimal_child.value))
				
			# if _ > 4:
				# print(self.history.children[self.actions.index((0,0,1))].value)
				# print(self.history.children[self.actions.index((0,0,1))].visited)

		optimal_action = self.history.optimal_action(0)
		optimal_child = self.history.children[self.actions.index(optimal_action)]

		print((optimal_action, optimal_child.value))

		l = []
		for child in self.history.children:
			l.append(child.value)
		print(l)

		#print(optimal_child.visited)
		
		# # #THIS TEST CASE NOW WORKS!!
		#for action in self.history.children:

		# # #THIS TEST CASE DOES NOT - also test visited of right below optimal child - wait it does now??!
		# for action in optimal_child.children[0].children:
		# 	print(action.visited)
		# 	t = 0
		# 	for child in action.children:
		# 		for i in range(0, len(self.theta_list)):
		# 			t = t + child.visited_list[i]
		# 	print(t)
		# 	print(" ")
		# 	t = 0
		return self.data

	def random_sample(self, list_to_sample):
		"""
		Returns a random sample from the list_to_sample.
		:param list_to_sample: the list we want to sample from
		"""
		random_index = np.random.choice(range(0, len(list_to_sample)))
		return list_to_sample[random_index]

	def rollout_helper(self, state, robot_action, history, depth):
		"""
		Calls the rollout function and adds the search nodes created by the rollout to search tree.
		Returns the value from the rollout function.
		:param state: the starting rollout state
		:param robot_action: the starting robot_action (the original optimal_action)
		:param history: the history from which we begin the rollout (we add children to this history)
		:param depth: the current depth we are at in the search tree
		"""
		random_human_action = self.random_sample(self.observations)
		value = self.rollout(state, robot_action, random_human_action, depth)

		#check this.
		new_robot_action_child = RobotNode(self.game, value, 1)
		new_human_obs_child = HumanNode(self.game, value, state[1])
		new_robot_action_child.children[self.observations.index(random_human_action)] = new_human_obs_child
		history.children[self.actions.index(robot_action)] = new_robot_action_child

		#these should be here.
		history.update_visited(state[1])
		history.update_value(value, state[1])

		return value

	def rollout(self, state, robot_action, human_action, depth):
		"""
		Actually performs the rollout by randomly sampling robot and human actions.
		Returns the value achieved from the rollout.
		:param robot_action: the robot action to be used for the rollout
		:param human_action: the human action to be sued for the rollout
		"""
		if math.pow(self.gamma, depth) < self.epsilon:
			return 0

		if self.game.getReward(state):
			return 1

		next_state = self.game.getNextState(state, robot_action, human_action)
		next_robot_action = self.random_sample(self.actions)
		next_human_action = self.random_sample(self.observations)

		#this should multiply the gamma here right?
		return self.gamma*self.rollout(next_state, next_robot_action, next_human_action, depth + 1)

	def simulate(self, state, history, depth):
		"""
		The recursive Simulate function as described in Silver et al. Simulates the start state moving down
		the search tree, by picking the optimal action at each point in the tree and simulating
		future observations (human actions). Incrementally builds the search tree and updates
		the values/visited counts of each search node it hits. Returns the value achieved from
		passing the state down.
		:param state: the starting state
		:param history: the history we are currently at in the search tree
		:param depth: the current depth we are at in the search tree
		"""
		if math.pow(self.gamma, depth) < self.epsilon:
			return 0
			
		if self.game.getReward(state):
			history.update_visited(state[1])
			history.update_value(1, state[1])
			return 1

		optimal_action = history.optimal_action(self.c)
		
		if history.children[self.actions.index(optimal_action)] == "empty":
			rollout_value = self.rollout_helper(state, optimal_action, history, depth)
			#should this be the reward or gamma^depth*reward
			return rollout_value

		if self.behavior == "boltzmann":
			next_state, human_action = self.sampleBoltzmann(state, optimal_action, history)
		elif self.behavior == "rational":
			next_state, human_action = self.sampleRational(state, optimal_action, history)
		else:
			print("wrong behavior input")

		next_history = history.children[self.actions.index(optimal_action)].children[self.observations.index(human_action)]
		if next_history == "empty":
			
			#what the fuck is this hahaha?
			new_human_obs_child = new_human_obs_child = HumanNode(self.game, 0, state[1])
			new_human_obs_child.visited_list[self.theta_list.index(state[1])] -= 1
			history.children[self.actions.index(optimal_action)].children[self.observations.index(human_action)] = new_human_obs_child
			next_history = history.children[self.actions.index(optimal_action)].children[self.observations.index(human_action)]

		R = self.gamma*self.simulate(next_state, next_history, depth + 1)

		history.children[self.actions.index(optimal_action)].update_visited()
		history.children[self.actions.index(optimal_action)].update_value(R)

		#I THINK this should be history and not next_history. Otherwise overcount during rollouts.
		history.update_visited(state[1])
		history.update_value(R, state[1])

		return R

	def sampleBoltzmann(self, state, robot_action, history):
		"""
		Places a Boltzmann distribution over the future observations (human actions).
		Returns a sampled human action from this distribution.
		:param state: the state we are currently at
		:param robot_action: the robot action that was selected to lead us to this set of human actions
		:param history: the part of search tree we are currently at
		"""
		theta = state[1]
		theta_index = self.theta_list.index(theta)
		qValues = []
		list_of_probabilities = []
		
		for child in history.children[self.actions.index(robot_action)].children:
			#add exploration bonus for 0 times human action for that theta HERE
			if child == "empty":
				qValues.append(1)
				#print("x")
			else:
				qValues.append(child.value_list[theta_index])

		#print(qValues)
		qValues = np.array(qValues)
		expQ_values = np.exp(self.beta * qValues)
		total = sum(expQ_values)
		
		for x in list(expQ_values):
			list_of_probabilities.append(x / total)

		random_index = np.random.choice(range(0, len(self.observations)), p = list_of_probabilities)
		random_human_action = self.observations[random_index]
		next_state = self.game.getNextState(state, robot_action, random_human_action)

		return next_state, random_human_action

	def sampleRational(self, state, robot_action, history):
		"""
		Assumes the human acts rationally with respect to his actions (+ exploration).
		Returns the rational human action.
		:param state: the state we are currently at
		:param robot_action: the robot action that was selected to lead us to this set of human actions
		:param history: the part of search tree we are currently at		
		"""
		theta = state[1]
		theta_index = self.theta_list.index(theta)
		qValues = []
		
		for child in history.children[self.actions.index(robot_action)].children:
			#add exploration bonus for 0 times human action for that theta HERE
			if child == "empty":
				qValues.append(0 + self.c)

				# OLD
				# child_index = history.children[self.actions.index(robot_action)].children.index(child)
				# next_human_action = self.observations[child_index]
				# next_state = self.game.getNextState(state, robot_action, next_human_action)
				# return next_state, next_human_action
				#print("x")
			else:
				qValues.append(child.value_list[theta_index] + self.c/(1.5*child.visited_list[theta_index] + 1))

		best_index = qValues.index(max(qValues))
		next_human_action = self.observations[best_index]
		next_state = self.game.getNextState(state, robot_action, next_human_action)
		return next_state, next_human_action

