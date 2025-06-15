import numpy as np
import pandas as pd

POSITIONS = ['C', 'F', 'G']
POSITION_LIMITS = {'C':2, 'F':4, 'G':4}
BUDGET = 100.0

class Selection:
    def __init__(self, selection_df: pd.DataFrame, full_roster_df: pd.DataFrame) -> None:
        self.selection_df = selection_df
        self.full_roster_df = full_roster_df

    def __str__(self):
        return str(self.selection_df)
    
    def copy(self):
        return Selection(self.selection_df, self.full_roster_df)

    def total_fp(self):
        return self.selection_df['fp'].sum()

    def total_dv(self):
        return self.selection_df['dv'].sum()
    
    def is_valid(self):
        position_counts = self.selection_df.value_counts('position').reindex(POSITIONS, fill_value=0)
        valid_positions = all(position_counts[pos] == POSITION_LIMITS[pos] for pos in POSITIONS)
        team_counts = self.selection_df.value_counts('team')
        valid_teams = sum([1 if count > 2 else 0 for count in team_counts.values]) == 0
        valid_dv = self.total_dv() <= BUDGET
        return valid_positions and valid_teams and valid_dv

    def possible_swaps(self):
        all_teams = self.full_roster_df['team'].unique()
        available_players_df = self.full_roster_df[~self.full_roster_df['id'].isin(self.selection_df['id'])]
        team_counts = self.selection_df.value_counts('team')
        maxed_teams = [team for team, count in team_counts.items() if count == 2]
        current_dv = self.total_dv()
        all_swaps = []
        for _, player_data in self.selection_df.iterrows():
            swap_position = player_data.loc['position']
            swap_teams = [team for team in all_teams if team == player_data.loc['team'] or not team in maxed_teams]
            swap_budget = BUDGET - current_dv + player_data.loc['dv']
            alternate_players_mask = \
                (available_players_df['team'].isin(swap_teams)) & \
                (available_players_df['dv'] <= swap_budget) & \
                (available_players_df['position'] == swap_position)
            alternate_players_df = available_players_df[alternate_players_mask]
            swap_list = [(player_data.loc['name'], alternate_player_data.loc['name']) for _, alternate_player_data in alternate_players_df.iterrows()]
            all_swaps.extend(swap_list)
        return(np.array(all_swaps))
    
    def possible_additions(self):
        available_players_df = self.full_roster_df[~self.full_roster_df['id'].isin(self.selection_df['id'])]
        #create list of possible teams (teams with less than 2 teams on the current roster)
        all_teams = self.full_roster_df['team'].unique()
        team_counts = self.selection_df.value_counts('team')
        maxed_teams = [team for team, count in team_counts.items() if count == 2]
        possible_teams = [team for team in all_teams if team not in maxed_teams]
        #determine remaining budget
        possible_budget = BUDGET - self.total_dv()
        #create list of possible positions (positions which appear less than their limit)
        position_counts = self.selection_df.value_counts('position').reindex(POSITIONS, fill_value=0)
        possible_positions = [pos for pos in POSITIONS if position_counts[pos] < POSITION_LIMITS[pos]]
        possible_players_mask = \
            (available_players_df['team'].isin(possible_teams)) & \
            (available_players_df['dv'] <= possible_budget) & \
            (available_players_df['position'].isin(possible_positions))
        # maybe just return names to make it more consistent with possible swaps?
        return available_players_df[possible_players_mask]
        


    def add_player(self, name) -> None:
        new_player_data = self.full_roster_df[self.full_roster_df['name'] == name]
        #add exception if player isn't in the full roster?
        self.selection_df = pd.concat([self.selection_df, new_player_data])

    def remove_player(self, name) -> None:
        self.selection_df = self.selection_df[~(self.selection_df['name'] == name)]

    def swap_players(self, swap) -> None:
        self.remove_player(swap[0])
        self.add_player(swap[1])

    
