import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define a function to simulate the remainder of the season.
def simulate_player_performance(home_runs_so_far, sb_so_far, games_played_so_far, total_games=162, simulations=1000000, career_sb=86, career_home_runs=171, career_games=701):
    
    remaining_games = total_games - games_played_so_far

    bayes_strength = 0.05
    alpha_hr_prior = bayes_strength * career_home_runs
    alpha_sb_prior = bayes_strength * career_sb
    beta_prior = bayes_strength * career_games

    alpha_hr_posterior = alpha_hr_prior + home_runs_so_far
    alpha_sb_posterior = alpha_sb_prior + sb_so_far
    beta_posterior = beta_prior + games_played_so_far

    lambdas_hr = np.random.gamma(alpha_hr_posterior, 1 / beta_posterior, simulations)
    lambdas_sb = np.random.gamma(alpha_sb_posterior, 1 / beta_posterior, simulations)

    simulated_home_runs = np.random.poisson(lambdas_hr * remaining_games)
    simulated_sb = np.random.poisson(lambdas_sb * remaining_games)
    
    return simulated_home_runs, simulated_sb

# Load Ohtani's stats
ohtani_stats = pd.read_csv('Ohtani_2024_Game_Log.csv')
ohtani_stats['Date'] = pd.to_datetime(ohtani_stats['Date'])
ohtani_stats['Cumulative_HR'] = ohtani_stats['HR'].cumsum()
ohtani_stats['Cumulative_SB'] = ohtani_stats['SB'].cumsum()
ohtani_stats['Game_Count'] = np.arange(1, len(ohtani_stats)+1)

# Calculate remaining games
total_games = 162
games_played = len(ohtani_stats)
remaining_games = total_games - games_played

# Create a list to store probabilities
probabilities_50_50 = []
home_runs_so_far = ohtani_stats['Cumulative_HR'].iloc[-1]
sb_so_far = ohtani_stats['Cumulative_SB'].iloc[-1]
games_played_so_far = ohtani_stats['Game_Count'].iloc[-1]

cumulative = 0
# Simulate for each of the remaining games
for i in range(remaining_games):
    # Stats after the current game
    game_no = games_played_so_far + i + 1

    # Simulate the remainder of the season
    simulated_home_runs, simulated_sb = simulate_player_performance(home_runs_so_far, sb_so_far, games_played_so_far, total_games=game_no)

    # Calculate probability of reaching 50-50
    prob_50_50 = np.mean((simulated_home_runs + home_runs_so_far >= 50) & (simulated_sb + sb_so_far >= 50)) * 100 - cumulative
    cumulative += prob_50_50
    probabilities_50_50.append(prob_50_50)

games_remaining = ['9/13','9/14','9/15','9/16','9/17','9/18','9/19','9/20','9/21','9/22','9/24','9/25','9/26','9/27','9/28','9/29']

opponent_colors = ['#CE1141','#CE1141','#CE1141','#CE1141','#00A3E0','#00A3E0','#00A3E0','#333366','#333366','#333366','#2F241D','#2F241D','#2F241D','#333366','#333366','#333366']

# Plotting
plt.figure(figsize=(10, 6))
bars = plt.bar(range(games_played_so_far+1, 163), probabilities_50_50)
for i in range(len(bars)):
    bars[i].set_color(opponent_colors[i])
ax = plt.subplot()
ax.set_xticks(range(games_played_so_far+1, 163))
ax.set_xticklabels(games_remaining)
plt.xlabel('Game Date')
plt.ylabel('Probability (%)')
plt.ylim(0,10)
plt.title('When is Shohei Ohtani going to reach 50-50?')
#plt.grid(True)
plt.show()
