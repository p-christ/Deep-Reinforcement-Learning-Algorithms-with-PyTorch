# not sure how this is different to A2C

# Initialise s and theta
#
# for timesteps:
#     Sample an action from policy
#     Sample r and s at t+1
#
#     Calculate td-error / advantage:  r_(t+1) + gamma * v(s_(t+1)) - v(s_t)
#
#     w <-- w + learning_rate * td_error * gradient of value network
#     theta <-- theta + learning_rate * td_error * gradient of log probability of action

#