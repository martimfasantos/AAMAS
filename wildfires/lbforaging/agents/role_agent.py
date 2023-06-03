import numpy as np
from . import Agent
from ..foraging.environment import Action
from ..foraging import FireTruck

class RoleAgent(Agent):
    name = "Role Agent"

    def __init__(self, player, roles):
        super().__init__(player)
        self.target_fire = None
        self.target_water_resource = None
        self.roles = roles
        self.role_assignment = None
        self.role_assignment_period = 10
        self.currentStep = 0
        self.localSteps = 0

    def potential_function(self, agent, fire):
        """
        Calculates the potential function used for role assignment.
        """
        raise NotImplemented("Potential Functions are implemented in the subclasses")

    def role_assignemt(self, obs):
        total_fire_lvl = [sum([fire.level for fire in fires]) for fires in obs.fires]
        
        # Calculate potentials for all agents and roles.
        potentials = np.zeros((len(obs.fires), len(obs.players)))
        for i, fire in enumerate(obs.fires):
            for j, agent in enumerate(obs.players):
                potentials[i, j] = self.potential_function(agent, fire, role)
        
        agents_roles = np.zeros(len(obs.players), dtype=np.int32)
        assigned_water = np.zeros(len(obs.fires), dtype=np.int32)

        # Assign roles to agents: example ROLES = [FIRE, FIRE, FIRE, WATER]
        # TODO: FIX THIS Check this assignments
        for (role_idx, role) in enumerate(self.roles):
            agent_id = np.argmax(potentials[role_idx, :])
            fire_id = np.argmax(potentials[:,i])
            agents_roles[i] = fire_id
            assigned_water[fire_id] += obs.players[i].controller.water

            if(assigned_water[fire_id] >= total_fire_lvl[fire_id]): # fire will be extinguished
                potentials[fire_id,:] = -999
           
        self.role_assignment = agents_roles
    
    @staticmethod
    def manhattan_distance(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def step(self, obs):
        self.localSteps += 1
        self.currentStep = self.localSteps 
        # assign the roles if it is time
        if(self.role_assignment is None or self.currentStep % self.role_assignment_period == 0):
            self.assignRoles(obs)

        if(self.water == 0):
            return self._refill_water(obs)
        else:

            # get the assinged fire
            assignedFire = obs.fires[self.role_assignment[self.id]]

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
    name = "R1"

    """
    The fire's roles are assigned based on the distance to the closest fire tile
    """

    def potential_function(self, agent, fire):
        return -min([self.manhattan_distance(agent.observed_position,(fire.row,fire.col)) for fire in fire])
    

class R2(RoleAgent):
    name = "R2"

    """
    The fire's roles are assigned based on how much of that fire an agent can extinguish
    """

    def potential_function(self, agent, fire):
        return agent.controller.water - sum([fire.level for fire in fire]) * 100

    
# TODO: Implement this
class R3(RoleAgent):
    name = "R3"

    """
    The fire's roles are assigned based on how much of that fire an agent can extinguish and the distance to it
    """

    def potential_function(self,agent,fire):
        pass

