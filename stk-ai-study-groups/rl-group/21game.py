import numpy as np

class Game21():
    def __init__(self):
        
        pass

    def start_game(self, player0, player1):
        self.players = []
        self.players.append(player0)
        self.players.append(player1)
        self.current_number = 21

        self.current_player_idx = np.random.randint(0, 2)
        initial_idx = self.current_player_idx
        while (self.current_number > 0):
            action = self.players[self.current_player_idx].get_action(self.current_number)
            #print("game:{0} -> player{1}, ac:{2}".format(self.current_number, self.current_player_idx, action))
            self.current_number -= action

            #next turn
            self.current_player_idx = self._get_next_player_idx(self.current_player_idx)

        winner_idx = self.current_player_idx
        return 1 if winner_idx == initial_idx else 0 #return winner
        

    def _get_next_player_idx(self, current_idx):
        return (current_idx + 1) % 2

        
class RandomPlayer():
    def __init__(self):
        pass

    def get_action(self, current_number):
        max_possible_action = min(current_number, 4)
        return np.random.randint(1, max_possible_action + 1)

class PerfectPlayer():
    def __init__(self):
        pass

    def get_action(self, current_number):
        best_possible_action = (current_number % 5)
        if (best_possible_action == 0):
            best_possible_action = 5

        best_possible_action -= 1
        best_possible_action = max(best_possible_action, 1)
        #print("curr:{0}, ac:{1}".format(current_number, best_possible_action))
        return best_possible_action


def main():
    game = Game21()
    ply_0 = PerfectPlayer()
    ply_1 = PerfectPlayer()

    result = 0 
    num_games = 10000
    for i in range(num_games):
        result += game.start_game(ply_0, ply_1)

    print(result/num_games)


if __name__ == "__main__":
    main()