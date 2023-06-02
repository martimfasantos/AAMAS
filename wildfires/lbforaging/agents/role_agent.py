import numpy as np
from . import Agent
from ..foraging.environment import Action
from ..foraging import FireTruck
from .heuristic_agent import HeuristicAgent

class RoleAgent(HeuristicAgent):
    name = "Role Agent"

    roleAssignment = None
    roleAssignmentPeriod = 10
    currentStep = 0


    def __init__(self, player):
        super().__init__(player)
        self.localSteps = 0

    def potentialFunction(self,agent,fire):
        raise NotImplemented("Potential Functions are implemented in the subclasses")

    def assignRoles(self,obs):
        potentials = np.zeros((len(obs.fires),len(obs.players)))
        totalFireLevel = [sum([fire.level for fire in fires])*100 for fires in obs.fires]

        for i,fire in enumerate(obs.fires):
            for j,agent in enumerate(obs.players):
                potentials[i,j] = self.potentialFunction(agent,fire)
        

        agents_roles = np.zeros(len(obs.players), dtype=np.int32)
        assignedWater = np.zeros(len(obs.fires), dtype=np.int32)

        # TODO: Check this assignments
        for i in range(len(obs.players)):
            fire_id = np.argmax(potentials[:,i])
            agents_roles[i] = fire_id
            assignedWater[fire_id] += obs.players[i].controller.water

            if(assignedWater[fire_id] >= totalFireLevel[fire_id]): # fire will be extinguished
                potentials[fire_id,:] = -999
           
        self.roleAssignment = agents_roles
    
    @staticmethod
    def manhattanDistance(a,b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def step(self, obs):
        self.localSteps += 1
        self.currentStep = self.localSteps 
        # assign the roles if it is time
        if(self.roleAssignment is None or self.currentStep % self.roleAssignmentPeriod == 0):
            self.assignRoles(obs)

        if(self.water == 0):
            return self._refill_water(obs)
        else:

            # get the assinged fire
            assignedFire = obs.fires[self.roleAssignment[self.id]]

            y,x = self.observed_position

            minDistance = 1000000
            target = (0,0)

            # find the closest tile and move towards it
            for fire in assignedFire:
                frow,fcol = fire.row,fire.col
                dis = abs(frow - y)  + abs(fcol - x)
                if abs(frow - y)  + abs(fcol - x) == 1:
                    return Action.EXTINGUISH
                
                if(dis < minDistance):
                    minDistance = dis
                    target = (frow,fcol)

            return self._move_towards(target, obs.actions)


class R1(RoleAgent):
    """
    The fire's roles are assigned based on the distance to the closest fire tile
    """
    name = "R1"

    def potentialFunction(self,agent,fire):
        return -min([self.manhattanDistance(agent.observed_position,(fire.row,fire.col)) for fire in fire])
    


class R2(RoleAgent):
    """
    The fire's roles are assigned based on how much of that fire an agent can extinguish
    """
    name = "R2"

    def potentialFunction(self,agent,fire):
        return agent.controller.water - sum([fire.level for fire in fire]) * 100
    
# TODO: Implement this
class R3(RoleAgent):
    """
    The fire's roles are assigned based on how much of that fire an agent can extinguish and the distance to it
    """
    name = "R3"

    def potentialFunction(self,agent,fire):
        pass

