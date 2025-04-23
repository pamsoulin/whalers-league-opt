import numpy as np
import pandas as pd

class Selection:
    def __init__(self, selection_df: pd.DataFrame, full_roster_df: pd.DataFrame) -> None:
        self.selection_df = selection_df
        self.full_roster_df = full_roster_df

    def __str__(self):
        return str(self.selection_df)

    def total_fp(self):
        return self.selection_df['fp'].sum()

    def total_dv(self):
        return self.selection_df['dv'].sum()

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
            swap_budget = 100.0 - current_dv + player_data.loc['fp']
            alternate_players_mask = \
                (available_players_df['team'].isin(swap_teams)) & \
                (available_players_df['dv'] <= swap_budget) & \
                (available_players_df['position'] == swap_position)
            alternate_players_df = available_players_df[alternate_players_mask]
            swap_list = [(player_data.loc['name'], alternate_player_data.loc['name']) for _, alternate_player_data in alternate_players_df.iterrows()]
            all_swaps.extend(swap_list)
        #print(all_swaps)
        return(np.array(all_swaps))


    def swap_players(self, swap):
        #old_player_data = self.full_roster_df[self.full_roster_df['name'] == swap[0]]
        new_player_data = self.full_roster_df[self.full_roster_df['name'] == swap[1]]
        # print(new_player_data)
        # print(self.selection_df)
        new_selection_df = pd.concat([self.selection_df[~(self.selection_df['name']==swap[0])], new_player_data])
        return Selection(new_selection_df, self.full_roster_df)
