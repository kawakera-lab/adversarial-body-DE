import gym
import pprint
import numpy as np
from scipy.optimize import differential_evolution
import make_adv_length_ant
import datetime
import csv
from play import Play
from agent import Agent
import requests

name_list = ["Ant"]

num_size_value = {"Humanoid":19,"Ant":13,"Walker2d":7,"A1":20}
num_length_value = {"Humanoid":16,"Ant":12,"Walker2d":7,"A1":12} 
k = 0
for name in name_list:
	print("repeat:",k)
	ENV_NAME = name
	num_eval = 1 # 50
	maxiteration = 1 # 100
	pop_size = 1 # 8

	initial_size = []
	initial_length = []
	for i in range(num_size_value[name]):
		initial_size.append(1)
	for i in range(num_length_value[name]):
		initial_length.append(1)	

	if ENV_NAME == "Ant":
		make_adv_length_ant.set_adv_length(initial_length)

	bounds = []
	length_epsilon = 0 
	
	if k == 0:
		size_epsilon = 0.0
		length_epsilon = 0.05
		k = 1
			
	if size_epsilon!=0 and length_epsilon!=0:
		for i in range(num_size_value[ENV_NAME]): 
			bounds.append((1-size_epsilon,1+size_epsilon))	
		for i in range(num_length_value[ENV_NAME]): 
			bounds.append((1-length_epsilon,1+length_epsilon))
			
	elif size_epsilon==0:
		for i in range(num_length_value[ENV_NAME]): 
			bounds.append((1-length_epsilon,1+length_epsilon))

	elif length_epsilon==0:
		for i in range(num_size_value[ENV_NAME]):
			bounds.append((1-size_epsilon,1+size_epsilon))
			
					
	test_env = gym.make(ENV_NAME +"-v2")
	n_states = test_env.observation_space.shape[0]
	action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
	n_actions = test_env.action_space.shape[0]
	n_iterations = 2000
	lr = 3e-5
	device = "cpu"

	agent = Agent(n_states=n_states,
		        n_iter=n_iterations,
		        env_name=ENV_NAME,
		        action_bounds=action_bounds,
		        n_actions=n_actions,
		        lr=lr,
		        device=device)   
		        
	def make_adv_length(ENV_NAME,ratio_length):
		if ENV_NAME == "Ant":
			make_adv_length_ant.set_adv_length(ratio_length)

	def func_size_length(ratio):
		sum_reward = 0	
		ratio_size = ratio[0:num_size_value[ENV_NAME]] 
		ratio_length = ratio[num_size_value[ENV_NAME]:]	
		make_adv_size(ENV_NAME,ratio_size) 
		make_adv_length(ENV_NAME,ratio_length)	
		
		env = gym.make(ENV_NAME + "-v2")		
		for j in range(num_eval):
			player = Play(env, agent, ENV_NAME) 
			sum_reward += player.evaluate()
		avr_reward = sum_reward/num_eval
		
		return avr_reward

	def func_size(ratio):
		sum_reward = 0	
		ratio_size = ratio[0:num_size_value[ENV_NAME]]
		make_adv_size(ENV_NAME,ratio_size)
		
		env = gym.make(ENV_NAME + "-v2")	
		#average	
		for j in range(num_eval):
			player = Play(env, agent, ENV_NAME) 
			sum_reward += player.evaluate()
		avr_reward = sum_reward/num_eval
		
		return avr_reward
	
	def func_length(ratio):
		sum_reward = 0	
		ratio_length = ratio[0:num_length_value[ENV_NAME]]#	
		make_adv_length(ENV_NAME,ratio_length)
		
		env = gym.make(ENV_NAME + "-v2")		
		#average
		for j in range(num_eval):
			player = Play(env, agent, ENV_NAME) 
			sum_reward += player.evaluate()
		avr_reward = sum_reward/num_eval
		
		return avr_reward

	dt_now = datetime.datetime.now()
	old_day = dt_now.day
	old_hour = dt_now.hour
	old_minute = dt_now.minute
	old_second = dt_now.second
	time_info = " "+str(dt_now.month) +"."+ str(dt_now.day) +"."+ str(dt_now.hour) +":"+ str(dt_now.minute)+""+ str(dt_now.second)
	
	with open("./attack/differential_evolution/"+"diff_evo"+time_info+".csv", 'a') as f:
	    writer = csv.writer(f)
	    writer.writerow(["iteration","time","reward"])
	
	if size_epsilon==0:
		result = differential_evolution(func_length, bounds,epsilon=length_epsilon,timeinfo=time_info,maxiter=maxiteration,popsize=pop_size,disp=True)
		
	print(result)
	
	dt_now = datetime.datetime.now()
	new_day = dt_now.day
	new_hour = dt_now.hour
	new_minute = dt_now.minute
	new_second = dt_now.second
	dlt_time = [0,0,0,0]
	dlt_time[0] = new_day - old_day
	dlt_time[1] = new_hour - old_hour
	dlt_time[2] = new_minute - old_minute
	dlt_time[3] = new_second - old_second
	for i in range(1,4):
		if dlt_time[i] < 0:
			dlt_time[i-1] = dlt_time[i-1] - 1
			if i == 1:
				dlt_time[i] = dlt_time[i] + 24
			else:                	
				dlt_time[i] = dlt_time[i] + 60 


	path = "./attack/differential_evolution/"+"diff_evo"+time_info+".txt"

	f = open(path,"a")
	f.write("\n\n")
	f.writelines(ENV_NAME+" size_eps:"+str(size_epsilon)+" length_eps:"+str(length_epsilon)+" maxiter:"+str(maxiteration)+" num_eval:"+str(num_eval)+" pop_size:"+str(pop_size)) # size もいれたい
	f.write("\n")
	f.write(str(result))
	f.write("\n")
	f.writelines("pasted day:"+str(dlt_time[0])+" hour:"+str(dlt_time[1])+" min:"+str(dlt_time[2])+ " second:"+str(dlt_time[3]))
	f.write("\n\n")


