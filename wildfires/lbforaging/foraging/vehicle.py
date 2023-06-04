import logging

class Vehicle():
    """
    Wrapper around an agent.
    """
    name = "Vehicle Agent"

    def __repr__(self):
        return self.name

    def __getattr__(self, item):
        return getattr(self.agent, item)

    def __init__(self, agent, water_capacity=None):
        self.logger = logging.getLogger(__name__)
        self.agent = agent
        self.water_capacity = water_capacity
        self.water = water_capacity # vehicles start with full water


    def step(self, obs):
        return self.agent._step(obs)
    
    def extinguish(self, fire_level):
        if self.water - fire_level * 100 < 0:
            extinguised_level = self.water // 100
            self.water = 0
            return extinguised_level
        else:
            self.water -= fire_level * 100
            return fire_level     

    def refill(self, amount=None):
        self.water = amount if amount else self.water_capacity


class Helicopter(Vehicle):
    name = "Helicopter Agent"

    def __init__(self, agent):
        super().__init__(agent, water_capacity=200)


class FireTruck(Vehicle):
    name = "FireTruck Agent"

    def __init__(self, agent):
        super().__init__(agent, water_capacity=500)