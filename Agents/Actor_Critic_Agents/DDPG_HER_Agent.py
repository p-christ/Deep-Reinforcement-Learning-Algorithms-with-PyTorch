from DDPG_Agent import DDPG_Agent


# WIP not finished

#TODO override the save_experience and step method so that it only saves at end of episode and saves two types of experience,
#one with the normal goal and one as if the goal had been the state we achieved in the final timestep


class DDPG_HER_Agent(DDPG_Agent):
    agent_name = "DDPG_HER"

    def __init__(self, config):
        DDPG_Agent.__init__(self, config)

    def step(self):
        pass

    def save_experience(self):
        pass

