import numpy as np
from itertools import chain
from . import Agent
from ..foraging.environment import Action
from ..foraging import FireTruck

class RoleAgent(Agent):
    name = "Role Agent"

    def __init__(self, player):
        super().__init__(player)
        self.target_fire = None
        self.target_water_source = None
        self.role_assignment_period = 3 # TODO: review this OR make it a parameter
        self.role_assignments = None
        self.curr_role = None
        self.steps_counter = 0
        self.fire_levels = {}


    def potential_function(self, agent, fire):
        """
        Calculates the potential function used for role assignment.
        """
        raise NotImplemented("Potential Functions are implemented in the subclasses")

    def role_assignment(self, obs):

        potentials = np.zeros((len(obs.fires), len(obs.players)))
        total_fire_levels = [sum([fire.level for fire in fires]) for fires in obs.fires]
        print("Total fire levels: {}".format(total_fire_levels))

        for i, fire in enumerate(obs.fires):
            for j, agent in enumerate(obs.players):
                potentials[i, j] = self.potential_function(agent, fire)

        agents_roles = np.zeros(len(obs.players), dtype=np.int32)
        assigned_water = np.zeros(len(obs.fires), dtype=np.int32)

        # sort agent by level in descending order to improve efficiency
        agent_ids = np.arange(len(obs.players))
        agent_list = list(zip(agent_ids, obs.players))
        sorted_agents = sorted(agent_list, key=lambda x: x[1].level, reverse=True)

        print(sorted_agents)
        # Assign roles based on potential values and available water
        for id, agent in sorted_agents:
            fire_id = np.argmax(potentials[:, id])
            agents_roles[id] = fire_id
            assigned_water[fire_id] += obs.players[id].level
            print("Water assigned to fire {}: {}".format(fire_id, assigned_water[fire_id]))
            print("Player level: {}".format(obs.players[id].level))

            if assigned_water[fire_id] >= total_fire_levels[fire_id]:  # fire will be extinguished
                potentials[fire_id, :] = -999

        print(potentials)
        print("Agent roles: ", agents_roles)
        return agents_roles
    
    @staticmethod
    def manhattan_distance(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    
    def get_id(self, obs):
        for i, player in enumerate(obs.players):
            if player.is_self:
                return i
            
    def _compute_turn(self, direction, targeted_direction):
        turns = [Action.NONE, Action.TURN_RIGHT, Action.TURN_AROUND, Action.TURN_LEFT]
        return turns[(targeted_direction.value - direction.value) % 4]

    def _move_towards(self, target, allowed):

        y, x = self.observed_position
        r, c = target

        # self.forbidden prevents the agent from going back and forth between two tiles
        if r < y and Action.NORTH:
            if Action.NORTH in allowed:
                return Action.NORTH
            elif (isinstance(self.controller, FireTruck)):
                return self._compute_turn(self.direction, Action.NORTH)
        
        if r > y and Action.SOUTH:
            if Action.SOUTH in allowed:
                return Action.SOUTH
            elif (isinstance(self.controller, FireTruck)):
                return self._compute_turn(self.direction, Action.SOUTH)
            
        if c > x and Action.EAST:
            if (Action.EAST in allowed):
                return Action.EAST
            elif (isinstance(self.controller, FireTruck)):
                return self._compute_turn(self.direction, Action.EAST)
            
        if c < x and Action.WEST:
            if(Action.WEST in allowed):
                return Action.WEST
            elif(isinstance(self.controller, FireTruck)):
                return self._compute_turn(self.direction, Action.WEST)
            
        return Action.NONE
    
    
    def _refill_water(self, obs):
        
        y,x = self.observed_position
        r,c = self._closest_water_source(obs)

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.REFILL
        else:
            return self._move_towards((r, c), obs.actions)
    

    def step(self, obs):
        # assign the roles if it is time
        if self.curr_role is None or self.steps_counter % self.role_assignment_period == 0:
            self.role_assignments = self.role_assignment(obs)
            agent_id = self.get_id(obs)
            self.curr_role = self.role_assignments[agent_id]
            print("Agent: {} Target: {}".format(agent_id, self.curr_role))
        
        self.steps_counter += 1

        if self.water == 0:
            return self._refill_water(obs)
        else:
            agent_id = self.get_id(obs)
            assigned_fire = obs.fires[self.role_assignments[agent_id]]

        print("Assigned fire: {}".format(assigned_fire))
        if assigned_fire == []:
            return Action.NONE

        y, x = self.position

        minDistance = float('inf')
        target = (-1,-1)

        for fire in assigned_fire:
            frow, fcol = fire.row, fire.col
            dist = self.manhattan_distance((frow, fcol), (y, x))
            print(dist)
            if dist == 0 or dist == 1:
                return Action.EXTINGUISH
            
            if dist < minDistance:
                minDistance = dist
                target = (frow, fcol)
                print(target)

        return self._move_towards(target, obs.actions)

# TODO old implementation of R1
class R1(RoleAgent):
    name = "R1"

    """
    The fire's roles are assigned based on the distance to the closest fire tile
    """

    def potential_function(self, obs, agent, role):

        fire_distances = []
        for fire in [fire for fire_agreg in obs.fires for fire in fire_agreg]:
            distance = self.manhattan_distance(agent.position, (fire.row, fire.col))
            fire_distances.append((fire, distance))

        # Sort the fires by distance in ascending order
        fire_distances.sort(key=lambda x: x[1])
        self.closest_fire_level = fire_distances[0][0].level
        closest_fire = fire_distances[0][0]
        closest_fire_distance = fire_distances[0][1]

        closest_water_distance = float("inf")
        for ws in obs.water_sources:
            distance = self.manhattan_distance(agent.position, (ws.row, ws.col))
            if distance < closest_water_distance:
                closest_water_distance = distance
                closest_water = ws

        if role == "FIRE" and self.sum_lvl_assigned_closest < closest_fire.level:
            self.sum_lvl_assigned_closest += self.player.controller.water // 100
            return -closest_fire_distance
        if role == "FIRE" and self.sum_lvl_assigned_closest >= closest_fire.level:
            # assign to the secont closest fire
            return -self.manhattan_distance(agent.position, (self.target_fire.row, self.target_fire.col))
        elif self.player.controller.water == 0:
            if role == "WATER":
                return +999
            else:
                return -999
        else:
            if role == "FIRE":
                return -closest_fire_distance + self.player.controller.water // 100 + closest_fire.level + closest_water_distance
            else:
                return -closest_water_distance + 5 * self.player.controller.water_capacity / self.player.controller.water - closest_fire.level     

# TODO: Old implementation of R2 (not working)
class R2(RoleAgent):
    name = "R2"

    """
    The fire's roles are assigned based on how much of that fire an agent can extinguish
    """

    def _is_adjacent_fire(self, frow, fcol, row, col):
        if abs(frow - row) + abs(fcol - col) == 1:
            return True
        return False

    def potential_function(self, obs, agent, role):        
        # its a fire
        if (role.row, role.col) in self.fire_levels:
            # has already enough water to extinguish
            if self.fire_levels[(role.row, role.col)] == 0:
                return -999
            else:
                return -self.manhattan_distance(agent.position, (role.row, role.col))
        else:
            return -self.manhattan_distance(agent.position, (role.row, role.col))


class R3(RoleAgent):
    name = "R3"

    """
    The fire's roles are assigned based on the distance to the closest fire tile
    """

    def potential_function(self, agent, fire):
        min_dist = 999
        for f in fire:
            dist = self.manhattan_distance(agent.position, (f.row, f.col))
            if dist < min_dist:
                min_dist = dist
        return -min_dist


