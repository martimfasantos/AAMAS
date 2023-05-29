class Helicopter():
    """
    Wrapper around an agent.
    """
    name = "Helicopter Agent"

    def __init__(self,agent):
        self.agent = agent
        self.water = 200

    def step(self, obs):
        return self.agent._step(obs)
    
    
class FireTruck():
    """
    Wrapper around an agent.
    """
    name = "FireTruck Agent"

    def __init__(self,agent):
        self.agent = agent
        self.water = 500

    def step(self, obs):
        return self.agent._step(obs)