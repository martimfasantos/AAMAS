import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
import numpy as np
import random
from .vehicle import FireTruck, Helicopter

TILES_PER_FIRE = 4

MAX_FIRE_LEVEL = 7
MAX_WATER_DEPOSIT_SIZE = 500
MAX_PLAYER_LEVEL = MAX_WATER_DEPOSIT_SIZE // 100


class ExtinguishingMode(Enum):
    STRONGEST = 0
    WEAKEST = 1
    ANY = 3


class Action(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    NONE = 4
    EXTINGUISH = 5
    TURN_LEFT = 6
    TURN_RIGHT = 7
    TURN_AROUND = 8
    REFILL = 9


class CellEntity(Enum):
    # entity encodings for grid observations
    OUT_OF_BOUNDS = 0
    EMPTY = 1
    FIRE = 2
    AGENT = 3


class Player:
    def __init__(self):
        self.id = None
        self.controller = None
        self.position = None
        self.level = None
        self.field_size = None
        self.score = None
        self.reward = 0
        self.history = None
        self.current_step = None
        self.orietation = 0
        self.direction = Action.NORTH
        self.extinguishing_mode = ExtinguishingMode.ANY

    def setup(self, position, level, field_size, id):
        self.history = []
        self.position = position
        self.level = level
        self.field_size = field_size
        self.score = 0

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller.step(obs)
    
    def turn(self,angle):
        self.orietation = (self.orietation + angle) % 360
        self.direction = Action(self.orietation // 90)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"


class ForagingEnv(Env):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    metadata = {"render.modes": ["human"]}

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, \
                  Action.EXTINGUISH, Action.TURN_RIGHT, Action.TURN_LEFT, Action.TURN_AROUND,Action.REFILL]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step", "fires", "water_sources"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "level", "history", "reward", "is_self"]
    )  # reward is available only if is_self

    Fire = namedtuple("Fire", ["row", "col", "level"])
    WaterSource = namedtuple("WaterSource", ["row", "col", "level"])

    def __init__(
        self,
        players,
        field_size,
        max_fires,
        steps_incr,
        sight,
        max_episode_steps,
        force_coop,
        normalize_reward=True,
        grid_observation=False,
        penalty=0.0,
    ):
        self.logger = logging.getLogger(__name__)
        self.seed()
        self.players = [Player() for _ in range(players)]

        self.field = np.zeros(field_size, np.int32)

        self.penalty = penalty
        
        self.max_fires = max_fires
        self.steps_incr = steps_incr
        self.max_water_sources = 1 # TODO: make this a parameter
        self._fires_spawned = 0.0
        self.max_player_level = MAX_PLAYER_LEVEL
        self.sight = sight
        self.force_coop = force_coop
        self._game_over = None

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps

        self._normalize_reward = normalize_reward
        self._grid_observation = grid_observation

        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(6)] * len(self.players)))
        self.observation_space = gym.spaces.Tuple(tuple([self._get_observation_space()] * len(self.players)))

        self.viewer = None

        self.fires = []
        self.water_sources = []

        self.n_agents = len(self.players)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count
        """
        if not self._grid_observation:
            field_x = self.field.shape[1]
            field_y = self.field.shape[0]
            # field_size = field_x * field_y

            max_fires = self.max_fires
            max_fires_level = TILES_PER_FIRE * self.max_player_level * len(self.players)

            min_obs = [-1, -1, 0] * max_fires + [-1, -1, -1] * self.max_water_sources + [-1, -1, 0] * len(self.players)
            max_obs = [field_x-1, field_y-1, max_fires_level] * max_fires \
                + [field_x-1, field_y-1, 0] * self.max_water_sources + [
                field_x-1,
                field_y-1,
                self.max_player_level,
            ] * len(self.players)
        else:
            # grid observation space
            grid_shape = (1 + 2 * self.sight, 1 + 2 * self.sight)

            # agents layer: agent levels
            agents_min = np.zeros(grid_shape, dtype=np.float32)
            agents_max = np.ones(grid_shape, dtype=np.float32) * self.max_player_level

            # fires layer: fires level
            max_fires_level = self.max_player_level * len(self.players)
            fires_min = np.zeros(grid_shape, dtype=np.float32)
            fires_max = np.ones(grid_shape, dtype=np.float32) * max_fires_level

            # water sources layer: water source
            water_min = np.ones(grid_shape, dtype=np.float32) * -1
            water_max = np.zeros(grid_shape, dtype=np.float32)

            # access layer: i the cell available
            access_min = np.zeros(grid_shape, dtype=np.float32)
            access_max = np.ones(grid_shape, dtype=np.float32)

            # total layer
            min_obs = np.stack([agents_min, fires_min, water_min, access_min])
            max_obs = np.stack([agents_max, fires_max, water_max, access_max])

        return gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.float32)

    @classmethod
    def from_obs(cls, obs):
        players = []
        for p in obs.players:
            player = Player()
            player.setup(p.position, p.level, obs.field.shape)
            player.score = p.score if p.score else 0
            players.append(player)

        env = cls(players, None, None, None, None)
        env.field = np.copy(obs.field)
        env.current_step = obs.current_step
        env.sight = obs.sight
        env._gen_valid_moves()

        return env

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows),
                max(col - distance, 0) : min(col + distance + 1, self.cols),
            ]

        return (
            self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows), col
            ].sum()
            + self.field[
                row, max(col - distance, 0) : min(col + distance + 1, self.cols)
            ].sum()
        )

    def adjacent_fire(self, row, col):
        return (
            (self.field[row, col] if self.field[row, col] > 0 else 0)
            + (self.field[max(row - 1, 0), col] if self.field[max(row - 1, 0), col] > 0 else 0)
            + (self.field[min(row + 1, self.rows - 1), col] if self.field[min(row + 1, self.rows - 1), col] > 0 else 0)
            + (self.field[row, max(col - 1, 0)] if self.field[row, max(col - 1, 0)] > 0 else 0)
            + (self.field[row, min(col + 1, self.cols - 1)] if self.field[row, min(col + 1, self.cols - 1)] > 0 else 0)
        )

    def adjacent_fire_location(self, row, col, mode):
        if mode == ExtinguishingMode.ANY:
            if self.field[row, col] > 0:
                return row, col
            if row > 0 and self.field[row - 1, col] > 0:
                return row - 1, col
            elif row < self.rows - 1 and self.field[row + 1, col] > 0:
                return row + 1, col
            elif col > 0 and self.field[row, col - 1] > 0:
                return row, col - 1
            elif col < self.cols - 1 and self.field[row, col + 1] > 0:
                return row, col + 1
        elif mode == ExtinguishingMode.STRONGEST:
            strongest = -1
            strongest_location = None
            if self.field[row, col] > 0:
                stronger = self.field[row, col] > strongest
                strongest = self.field[row, col] if stronger else strongest
                strongest_location = (row, col) if stronger else strongest_location
            if row > 0 and self.field[row - 1, col] > 0:
                stronger = self.field[row - 1, col] > strongest
                strongest = self.field[row - 1, col] if stronger else strongest
                strongest_location = (row - 1, col) if stronger else strongest_location
            if row < self.rows - 1 and self.field[row + 1, col] > 0:
                stronger = self.field[row + 1, col] > strongest
                strongest = self.field[row + 1, col] if stronger else strongest
                strongest_location = (row + 1, col) if stronger else strongest_location
            if col > 0 and self.field[row, col - 1] > 0:
                stronger = self.field[row, col - 1] > strongest
                strongest = self.field[row, col - 1] if stronger else strongest
                strongest_location = (row, col - 1) if stronger else strongest_location
            if col < self.cols - 1 and self.field[row, col + 1] > 0:
                stronger = self.field[row, col + 1] > strongest
                strongest = self.field[row, col + 1] if stronger else strongest
                strongest_location = (row, col + 1) if stronger else strongest_location
            return strongest_location
        elif mode == ExtinguishingMode.WEAKEST:
            weakest = np.iinfo(np.int32).max
            weakest_location = None
            if self.field[row, col] > 0:
                weaker = self.field[row, col] < weakest
                weakest = self.field[row, col] if weaker else weakest
                weakest_location = (row, col) if weaker else weakest_location
            if row > 0 and self.field[row - 1, col] > 0:
                weaker = self.field[row - 1, col] < weakest
                weakest = self.field[row - 1, col] if weaker else weakest
                weakest_location = (row - 1, col) if weaker else weakest_location
            if row < self.rows - 1 and self.field[row + 1, col] > 0:
                weaker = self.field[row + 1, col] < weakest
                weakest = self.field[row + 1, col] if weaker else weakest
                weakest_location = (row + 1, col) if weaker else weakest_location
            if col > 0 and self.field[row, col - 1] > 0:
                weaker = self.field[row, col - 1] < weakest
                weakest = self.field[row, col - 1] if weaker else weakest
                weakest_location = (row, col - 1) if weaker else weakest_location
            if col < self.cols - 1 and self.field[row, col + 1] > 0:
                weaker = self.field[row, col + 1] < weakest
                weakest = self.field[row, col + 1] if weaker else weakest
                weakest_location = (row, col + 1) if weaker else weakest_location
            return weakest_location
        
    def adjacent_water_source_location(self, row, col):
        if row > 1 and self.field[row - 1, col] == -1:
            return row - 1, col
        elif row < self.rows - 1 and self.field[row + 1, col] == -1:
            return row + 1, col
        elif col > 1 and self.field[row, col - 1] ==  -1:
            return row, col - 1
        elif col < self.cols - 1 and self.field[row, col + 1] == -1:
            return row, col + 1

    def adjacent_players(self, row, col):
        return [
            player
            for player in self.players
            if abs(player.position[0] - row) == 0
            and abs(player.position[1] - col) == 0
            or abs(player.position[0] - row) == 1
            and player.position[1] == col
            or abs(player.position[1] - col) == 1
            and player.position[0] == row
        ]

    def spawn_fires(self, max_fires, max_level):
        fire_count = 0
        attempts = 0
        min_level = max_level if self.force_coop else 1

        while fire_count < max_fires and attempts < 1000:
            attempts += 1
            row = self.np_random.integers(1, self.rows - 1)
            col = self.np_random.integers(1, self.cols - 1)

            # check if it has neighbors:
            if (
                self.neighborhood(row, col).sum() > 0
                or self.neighborhood(row, col, distance=2, ignore_diag=True) > 0
                or not self._is_empty_location(row, col)
            ):
                continue
            
            fire_level = max_level if self.force_coop else self.np_random.integers(min_level, max_level)
            self.field[row, col] = fire_level

            fire_count += 1 + self._fill_adjacent_tiles(row, col, fire_level)
        self._fires_spawned = self.field.sum()

    def spawn_water_sources(self, max_fires):
        water_sources = 0
        limit = self.np_random.integers(1, max(max_fires // (2 * TILES_PER_FIRE), 2))
        attempts = 0

        while water_sources < limit and attempts < 1000:
            attempts += 1
            row = self.np_random.integers(1, self.rows - 1)
            col = self.np_random.integers(1, self.cols - 1)

            # check if it has neighbors:
            if (
                self.neighborhood(row, col).sum() > 0
                or self.neighborhood(row, col, distance=2, ignore_diag=True) > 0
                or not self._is_empty_location(row, col)
            ):
                continue
            
            water_sources += 1
            self.field[row, col] = -1
            self.water_sources.append(self.WaterSource(row, col, self.field[row, col]))


    def _fill_adjacent_tiles(self, row, col, level):
        placed = 0
        fires = []
        fires.append(self.Fire(row, col, level))
        if (row - 1 >= 0 and self._is_empty_location(row-1, col)):
            self.field[row-1, col] = np.random.randint(1, level+1)
            fires.append(self.Fire(row-1, col, self.field[row-1, col]))
            placed += 1
        elif (row + 1 < self.rows and self._is_empty_location(row+1, col)):
            self.field[row+1, col] = np.random.randint(1, level+1)
            fires.append(self.Fire(row+1, col, self.field[row+1, col]))
            placed += 1
        if (col - 1 >= 0 and self._is_empty_location(row, col-1)):
            self.field[row, col-1] = np.random.randint(1, level+1)
            fires.append(self.Fire(row, col-1, self.field[row, col-1]))
            placed += 1
        elif (col + 1 < self.cols and self._is_empty_location(row, col+1)):
            self.field[row, col+1] = np.random.randint(1, level+1)
            fires.append(self.Fire(row, col+1, self.field[row, col+1]))
            placed += 1
        if (row - 1 >= 0 and col - 1 >=0 and self._is_empty_location(row-1, col-1)):
            self.field[row-1, col-1] = np.random.randint(1, level+1)
            fires.append(self.Fire(row-1, col-1, self.field[row-1, col-1]))
            placed += 1
        elif (row + 1 < self.rows and col + 1 < self.cols and self._is_empty_location(row+1, col+1)):
            self.field[row+1, col+1] = np.random.randint(1, level+1)
            fires.append(self.Fire(row+1, col+1, self.field[row+1, col+1]))
            placed += 1

        self.fires.append(fires)
        return placed
    
    def _is_empty_location(self, row, col):
        if self.field[row, col] != 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True

    def spawn_players(self, max_player_level, team):
        assert 'Helicopters' in team and 'Firetrucks' in team, \
            'Team must have a Helicopters and a FireTrucks definition'
        assert len(team['Firetrucks']) + len(team['Helicopters']) == len(self.players), \
            'Team must have the same number of FireTrucks and Helicopters as there are players'
        
        vehicles = team['Firetrucks'] + team['Helicopters']

        n_placed_players = 0
        for player in self.players:

            attempts = 0
            player.reward = 0

            while attempts < 1000:
                row = self.np_random.integers(0, self.rows)
                col = self.np_random.integers(0, self.cols)
                if self._is_empty_location(row, col):
                    agent = vehicles[n_placed_players]
                    if (n_placed_players < len(team['Firetrucks'])):
                        player.set_controller(FireTruck(agent(player)))
                    else:
                        player.set_controller(Helicopter(agent(player)))
                    player.setup(
                        (row, col),
                        player.controller.water // 100,
                        self.field_size,
                        n_placed_players
                    )
                    n_placed_players += 1
                    break
                attempts += 1

            
    def _check_direction(self,action, player):
       return isinstance(player.controller, Helicopter) or \
                (isinstance(player.controller, FireTruck) and action == player.direction)

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return player.position[0] > 0 and self._check_direction(action, player)
        elif action == Action.SOUTH:
            return player.position[0] < self.rows - 1 and self._check_direction(action, player)
        elif action == Action.WEST:
            return player.position[1] > 0 and self._check_direction(action, player)
        elif action == Action.EAST:
            return player.position[1] < self.cols - 1 and self._check_direction(action, player)
        elif action == Action.EXTINGUISH:
            return self.adjacent_fire(*player.position) > 0
        elif action == Action.TURN_RIGHT or \
             action == Action.TURN_LEFT or \
             action == Action.TURN_AROUND: # only trucks can turn
            return isinstance(player.controller, FireTruck)
        elif action == Action.REFILL:
            return self.adjacent_water_source_location(*player.position) and \
                player.controller.water < player.controller.water_capacity


        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player):
        return self.Observation(
            actions=self._valid_actions[player],
            players=[
                self.PlayerObservation(
                    position=self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    ),
                    level=a.level,
                    is_self=a == player,
                    history=a.history,
                    reward=a.reward if a == player else None,
                )
                for a in self.players
                if (
                    min(
                        self._transform_to_neighborhood(
                            player.position, self.sight, a.position
                        )
                    )
                    >= 0
                )
                and max(
                    self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    )
                )
                <= 2 * self.sight
            ],
            # todo also check max?
            field=np.copy(self.neighborhood(*player.position, self.sight)),
            game_over=self.game_over,
            sight=self.sight,
            current_step=self.current_step,
            fires=self._active_fires(),
            water_sources=self._water_sources(),
        )
    
    def _active_fires(self):
        fires = []
        for group in self.fires:
            fires.append([self.Fire(fire.row, fire.col, self.field[fire.row, fire.col]) 
                          for fire in group if self.field[fire.row, fire.col] > 0])
        self.fires = fires
        return self.fires
    
    def _water_sources(self):
        self.water_sources = [self.WaterSource(row, col, -1) for row, col in zip(*np.where(self.field == -1))]
        return self.water_sources
    
    def _increment_fires(self):
        for group in self._active_fires():
            for fire in group:
                new_fire_level = min(fire.level + 1, MAX_FIRE_LEVEL)
                fire = fire._replace(level = new_fire_level)
                self.field[fire.row, fire.col] = new_fire_level

    def _make_gym_obs(self):
        def make_obs_array(observation):
            obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
            # obs[: observation.field.size] = observation.field.flatten()
            # self player is always first
            seen_players = [p for p in observation.players if p.is_self] + [
                p for p in observation.players if not p.is_self
            ]

            for i in range(self.max_fires):
                obs[3 * i] = -1
                obs[3 * i + 1] = -1
                obs[3 * i + 2] = 0

            for i, (y, x) in enumerate(zip(*np.where(observation.field > 0))):
                obs[3 * i] = y
                obs[3 * i + 1] = x
                obs[3 * i + 2] = observation.field[y, x]

            for i in range(self.max_water_sources):
                obs[self.max_fires * 3 + 3 * i] = -1
                obs[self.max_fires * 3 + 3 * i + 1] = -1
                obs[self.max_fires * 3 + 3 * i + 2] = -1

            for i, (y, x) in enumerate(zip(*np.where(observation.field == -1))):
                obs[self.max_fires * 3 + 3 * i] = y
                obs[self.max_fires * 3 + 3 * i + 1] = x
                obs[self.max_fires * 3 + 3 * i + 2] = -1
            
            for i in range(len(self.players)):
                obs[(self.max_fires + self.max_water_sources) * 3 + 3 * i] = -1
                obs[(self.max_fires + self.max_water_sources) * 3 + 3 * i + 1] = -1
                obs[(self.max_fires + self.max_water_sources) * 3 + 3 * i + 2] = 0

            for i, p in enumerate(seen_players):
                obs[(self.max_fires + self.max_water_sources) * 3 + 3 * i] = p.position[0]
                obs[(self.max_fires + self.max_water_sources) * 3 + 3 * i + 1] = p.position[1]
                obs[(self.max_fires + self.max_water_sources) * 3 + 3 * i + 2] = p.level

            return obs

        def make_global_grid_arrays():
            """
            Create global arrays for grid observation space
            """
            grid_shape_x, grid_shape_y = self.field_size
            grid_shape_x += 2 * self.sight
            grid_shape_y += 2 * self.sight
            grid_shape = (grid_shape_x, grid_shape_y)

            agents_layer = np.zeros(grid_shape, dtype=np.float32)
            for player in self.players:
                player_x, player_y = player.position
                agents_layer[player_x + self.sight, player_y + self.sight] = player.level
            
            fires_layer = np.zeros(grid_shape, dtype=np.float32)
            fires_layer[self.sight:-self.sight, self.sight:-self.sight] = self.field.copy()

            waters_layer = np.zeros(grid_shape, dtype=np.float32)
            waters_layer[self.sight:-self.sight, self.sight:-self.sight] = self.field.copy()

            access_layer = np.ones(grid_shape, dtype=np.float32)
            # out of bounds not accessible
            access_layer[:self.sight, :] = 0.0
            access_layer[-self.sight:, :] = 0.0
            access_layer[:, :self.sight] = 0.0
            access_layer[:, -self.sight:] = 0.0
            # agent locations are not accessible
            for player in self.players:
                player_x, player_y = player.position
                access_layer[player_x + self.sight, player_y + self.sight] = 0.0
            # fire locations are not accessible
            fires_x, fires_y = self.field.nonzero() and self.field > 0
            for x, y in zip(fires_x, fires_y):
                access_layer[x + self.sight, y + self.sight] = 0.0
            # fire locations are not accessible
            waters_x, waters_y = self.field == -1
            for x, y in zip(waters_x, waters_y):
                access_layer[x + self.sight, y + self.sight] = 0.0
            
            return np.stack([agents_layer, fires_layer, waters_layer, access_layer])

        def get_agent_grid_bounds(agent_x, agent_y):
            return agent_x, agent_x + 2 * self.sight + 1, agent_y, agent_y + 2 * self.sight + 1
        
        def get_player_reward(observation):
            for p in observation.players:
                if p.is_self:
                    return p.reward

        observations = [self._make_obs(player) for player in self.players]
        if self._grid_observation:
            layers = make_global_grid_arrays()
            agents_bounds = [get_agent_grid_bounds(*player.position) for player in self.players]
            nobs = tuple([layers[:, start_x:end_x, start_y:end_y] for start_x, end_x, start_y, end_y in agents_bounds])
        else:
            nobs = tuple([make_obs_array(obs) for obs in observations])
        nreward = [get_player_reward(obs) for obs in observations]
        ndone = [obs.game_over for obs in observations]
        # ninfo = [{'observation': obs} for obs in observations]
        ninfo = {}
        
        # check the space of obs
        for i, obs in  enumerate(nobs):
            assert self.observation_space[i].contains(obs), \
                f"obs space error: obs: {obs}, obs_space: {self.observation_space[i]}"
        
        return observations, nreward, ndone, ninfo

    def reset(self, **kwargs):
        self.field = np.zeros(self.field_size, np.int32)
        self.spawn_players(self.max_player_level, kwargs['team'])
        # player_levels = sorted([player.level for player in self.players])

        self.spawn_fires(
            self.max_fires, max_level=MAX_FIRE_LEVEL
        )

        self.spawn_water_sources(self.max_fires)
        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()

        nobs, _, _, _ = self._make_gym_obs()
        return nobs, {}

    def step(self, actions):
        self.current_step += 1

        for p in self.players:
            p.reward = 0

        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

        # check if actions are valid
        for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info(
                    "{}{} attempted invalid action {}.".format(
                        player.name, player.position, action
                    )
                )
                actions[i] = Action.NONE

        extinguish_players = set()
        turning_right_players = set()
        turning_left_players = set()
        turning_around_players = set()

        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)
            elif action == Action.EXTINGUISH:
                collisions[player.position].append(player)
                extinguish_players.add(player)
            elif action == Action.TURN_RIGHT:
                collisions[player.position].append(player)
                turning_right_players.add(player)
            elif action == Action.TURN_LEFT:
                collisions[player.position].append(player)
                turning_left_players.add(player)
            elif action == Action.TURN_AROUND:
                collisions[player.position].append(player)
                turning_around_players.add(player)
            elif action == Action.REFILL:
                player.controller.refill()
                player.level = player.controller.water // 100

        # and do movements for non colliding players
        for pos, players in collisions.items():
            for player in players:
                player.position = pos
                # process turnings
                if player in turning_right_players:
                    player.turn(90)
                elif player in turning_left_players:
                    player.turn(-90)
                elif player in turning_around_players:
                    player.turn(180)

        # finally process the extinguish actions:
        while extinguish_players:
            # find adjacent fire(s)
            player = extinguish_players.pop()
            res = self.adjacent_fire_location(*player.position, player.extinguishing_mode)
            if res is None:
                continue
            # get fire level
            frow, fcol = res
            fire_level = self.field[frow, fcol]

            # target fire is already extinguished
            if fire_level == 0:
                continue

            # adjacent players to the fire
            adj_players = self.adjacent_players(frow, fcol)
            adj_players = [
                p for p in adj_players if p in extinguish_players or p is player
            ]

            adj_player_level = sum([a.level for a in adj_players])

            extinguish_players = extinguish_players - set(adj_players)

            # print("adj_player_level: ", adj_player_level)
            # print("fire_level: ", fire_level)

            if adj_player_level < fire_level:
                # failed to extinguish the fire completely
                for a in adj_players:
                    a.reward -= self.penalty

            # else the fire is extinguised and each player scores points
            for a in adj_players:
                a.reward = float(a.level * fire_level)
                # update player level
                extinguished_level = a.controller.extinguish(fire_level)
                a.level = a.controller.water // 100
                # update fire level
                fire_level = max(0, fire_level - extinguished_level)
                self.field[frow, fcol] = fire_level
                if self._normalize_reward:
                    a.reward = a.reward / float(
                        sum(a.controller.water_capacity for a in adj_players) // 100 * self._fires_spawned
                    )  # normalize reward
                if fire_level == 0:
                    break

        field = np.copy(self.field)
        field[field == -1] = 0
        self._game_over = (
            field.sum() == 0 or self._max_episode_steps <= self.current_step
        )
        # update fires
        if self.steps_incr is not None and self.current_step % self.steps_incr == 0:
            self._increment_fires()
        
        self._gen_valid_moves()

        for p in self.players:
            p.score += p.reward

        return self._make_gym_obs()

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
