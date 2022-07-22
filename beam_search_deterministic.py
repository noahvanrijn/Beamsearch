import numpy as np
import copy
import random
from queue import PriorityQueue

#CLASSES------------------------------------------------


class State:
    def __init__(self, games_left, forbidden_opponents, rounds, positions, home_games_streak, away_games_streak, total_distance, schedule):
        self.games_left = games_left
        self.forbidden_opponents = forbidden_opponents
        self.rounds = rounds
        self.positions = positions
        self.home_games_streak = home_games_streak
        self.away_games_streak = away_games_streak
        self.schedule = schedule
        self.total_distance = total_distance


# INSTANCES-------------------------------------------------------------
INSTANCE = 'nl6.txt'
beam_width = 6400
instances = np.loadtxt(INSTANCE)


#VARIABLES for root_state-------------------------------------
n = len(instances)
u = 3
layers = n ** 2 + n ** 2
games_left = np.ones((n, n), int)
np.fill_diagonal(games_left, 0)
forbidden_opponents = np.full((n,), -1, dtype=int)
rounds = np.zeros((n,), dtype=int)
positions = np.array(range(0, n))
home_games_streak = np.full((n), u)
away_games_streak = np.full((n), u)
schedule = np.zeros(shape=((2*n-2),n))
total_distance = np.zeros((n,), dtype=int)
no_repeat = True
#VARIABLES for terminal state--------------------------------------------------------------------
games_left_terminal = np.zeros((n, n), int)
forbidden_opponents_terminal = np.full((n,), -1, dtype=int)
rounds_terminal = np.ones((n,), dtype=int)
rounds_terminal = rounds_terminal * (2 * n - 1)
positions_terminal = np.array(range(1, n + 1))
home_games_streak_terminal = np.full(n, 0)
away_games_streak_terminal = np.full(n, 0)
#-----------------------------------------------------------------
root_state = State(games_left, forbidden_opponents, rounds, positions, home_games_streak, away_games_streak, total_distance, schedule)
terminal_state = State(games_left_terminal, forbidden_opponents_terminal, rounds_terminal, positions_terminal, home_games_streak_terminal, away_games_streak_terminal, total_distance, schedule)

#CONSTRAINTS-------------------------------------------------------------------------


def minimum_round_team(arr):
    item = np.min(arr)
    item_index, = np.where(arr == item)
    x = item_index[0]
    #print(x)

    return x


def minimum_round_array(arr, x):
    item = np.min(arr)
    item_index, = np.where(arr == item)
    team_array = np.array(x)
    minimum = np.setdiff1d(item_index, team_array)

    return minimum


def forbidden_opponents_constraint(arr, x, y):
    if arr[x] == y or arr[y] == x:
        return False
    else:
        return True


def at_most_constraint(arr, x):
    if arr[x] < 1:
        return False
    else:
        return True


def check_final_round(arr, state: State):
    final_round = 2 * n - 2
    list_of_values = []
    counter = 0
    for e in arr:
       if e == final_round:
          list_of_values.append(e)
       else:
          counter += 1

    if len(list_of_values) > 0 and np.array_equal(state.games_left, terminal_state.games_left):
        return True
    else:
        return False

#END CONSTRAINTS----------------------------------------------------------------------


def update_state(state: State, away_team, home_team):
    new_state = copy.deepcopy(state)

    # games left to play against each other
    state_games_left = new_state.games_left
    state_games_left[away_team, home_team] = 0

    # total number of rounds played
    state_rounds = new_state.rounds
    state_rounds[away_team] += 1
    state_rounds[home_team] += 1

    # save the old positions of the playing team
    state_old_positions = new_state.positions
    old_position_away = state_old_positions[away_team]
    old_position_home = state_old_positions[home_team]

    # new position of the playing teams
    state_positions = new_state.positions
    state_positions[away_team] = home_team
    state_positions[home_team] = home_team

    added_distance_away = instances[old_position_away, home_team]
    added_distance_home = instances[old_position_home, home_team]

    state_total_distance = new_state.total_distance
    state_total_distance[away_team] += added_distance_away
    state_total_distance[home_team] += added_distance_home

    state_home_games_streak = new_state.home_games_streak
    state_home_games_streak[away_team] = u
    state_home_games_streak[home_team] -= 1

    state_away_games_streak = new_state.away_games_streak
    state_away_games_streak[away_team] -= 1
    state_away_games_streak[home_team] = u

    # forbidden opponents
    state_forbidden_opponents = new_state.forbidden_opponents
    if no_repeat:
        if state_games_left[home_team, away_team]:
            state_forbidden_opponents[away_team] = home_team
            state_forbidden_opponents[home_team] = away_team
        else:
            state_forbidden_opponents[away_team] = -1
            state_forbidden_opponents[home_team] = -1

    round_in_schedule = state_rounds[away_team] - 1
    state_schedule = new_state.schedule
    state_schedule[round_in_schedule, away_team] = home_team
    state_schedule[round_in_schedule, home_team] = - away_team

    # total distance travelled so far

    # print(new_state.__dict__)
    return new_state


def update_state_to_terminal(state: State, away_team, home_team):
    new_state = copy.deepcopy(state)

    # games left to play against each other
    state_games_left = new_state.games_left

    # total number of rounds played
    state_rounds = new_state.rounds
    state_rounds[away_team] += 1
    state_rounds[home_team] += 1

    # save the old positions of the playing team
    state_old_positions = new_state.positions
    old_position_away = state_old_positions[away_team]
    old_position_home = state_old_positions[home_team]

    # new position of the playing teams
    state_positions = new_state.positions
    state_positions[away_team] = away_team
    state_positions[home_team] = home_team

    added_distance_away = instances[old_position_away, away_team]
    added_distance_home = instances[old_position_home, home_team]

    state_total_distance = new_state.total_distance
    state_total_distance[away_team] += added_distance_away
    state_total_distance[home_team] += added_distance_home

    state_home_games_streak = new_state.home_games_streak
    state_home_games_streak[away_team] = 0
    state_home_games_streak[home_team] = 0

    state_away_games_streak = new_state.away_games_streak
    state_away_games_streak[away_team] = 0
    state_away_games_streak[home_team] = 0

    # forbidden opponents
    state_forbidden_opponents = new_state.forbidden_opponents
    if no_repeat:
        if state_games_left[home_team, away_team]:
            state_forbidden_opponents[away_team] = home_team
            state_forbidden_opponents[home_team] = away_team
        else:
            state_forbidden_opponents[away_team] = -1
            state_forbidden_opponents[home_team] = -1

    return new_state


def create_first_layer(number_of_teams):
    teams = list(range(number_of_teams))
    x = teams.pop(0)
    print(x)

    layer_one = []
    for e in teams:
        state_away = copy.deepcopy(update_state(root_state, x, e))
        state_home = copy.deepcopy(update_state(root_state, e, x))
        layer_one.append(state_away)
        layer_one.append(state_home)

    return layer_one


def create_new_layer(layer):
    next_layer = []
    for node in layer:
        x = minimum_round_team(node.rounds)
        array_of_teams = minimum_round_array(node.rounds, x)
        no_solutions = 0

        for team in array_of_teams:
            if node.games_left[team, x] == 1 and forbidden_opponents_constraint(node.forbidden_opponents, x, team) and at_most_constraint(node.home_games_streak, x) and at_most_constraint(node.away_games_streak, team):
                state_home = copy.deepcopy(update_state(node, team, x))
                next_layer.append(state_home)
            if node.games_left[x, team] == 1 and forbidden_opponents_constraint(node.forbidden_opponents, x, team) and at_most_constraint(node.home_games_streak, team) and at_most_constraint(node.away_games_streak, x):
                state_away = copy.deepcopy(update_state(node, x, team))
                next_layer.append(state_away)
            if check_final_round(node.rounds, node):
                state_terminal = copy.deepcopy(update_state_to_terminal(node, x, team))
                next_layer.append(state_terminal)
            else:
                no_solutions += 1

    return next_layer


def beam_search():
    counter = 0
    next_layer = create_first_layer(n)

    while counter < layers:
        this_layer = create_new_layer(next_layer)
        #print(this_layer)
        # min heap
        pqueue = []
        for node in this_layer:
            total_travel_distance = sum(node.total_distance)
            pqueue.append((total_travel_distance, node))
            pqueue.sort(key=lambda tup: tup[0])

        top_beam_nodes = pqueue[0:beam_width]
        next_layer = []
        for e in top_beam_nodes:
            next_layer.append(e[1])

        #for state in next_layer:
            #print(state.__dict__)
            #total_distance = sum(state.total_distance)
            #print(total_distance)

        #print('layer')
        #print(counter)

        top_state = top_beam_nodes[0]
        print(top_state)
        print(top_state[0])
        state = top_state[1]
        print(state.__dict__)

        counter += 1


beam_search()




