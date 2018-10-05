from Environments.Unity_Environments.Reacher_Environment_1_Arm import Reacher_Environment_1_Arm

env = Reacher_Environment_1_Arm("/Users/petroschristodoulou/Documents/Deep_RL_Implementations/deep-reinforcement-learning/p2_continuous-control/Reacher.app")

print(env.get_action_size())
