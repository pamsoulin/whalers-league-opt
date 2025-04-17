import numpy as np
import pandas as pd

class Team:
    def __init__(self, team_df: pd.DataFrame) -> None:
        self.team_df = team_df

    def total_fp(self):
        return self.team_df['fp'].sum()

    def total_dv(self):
        return self.team_df['dv'].sum()

    def possible_swaps(self, full_roster_df):
        other_players_df = full_roster_df - self.team_df
        team_counts = self.team_df.value_counts('team')
        for player_data in self.team_df.iterrows():
            print(player_data)
            print(type(player_data))
            # player_pos = player_data.position
            # player_team = player_data.team
            # swaps_df = 
            # print(player_pos)