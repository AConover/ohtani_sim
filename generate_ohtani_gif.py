import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

#Define a function to simulate the remainder of the season.
def simulate_player_performance(home_runs_so_far, sb_so_far, games_played_so_far, total_games=162, simulations=1000000, career_sb = 86, career_home_runs=171, career_games=701):
    
    #Based on the number of games in the season, how many games are remaining
    remaining_games = total_games - games_played_so_far

    #Define the prior distribution, which is a Gamma. bayes_strength defines the strength of the prior for the update
    bayes_strength = 0.05
    alpha_hr_prior = bayes_strength*career_home_runs
    alpha_sb_prior = bayes_strength*career_sb
    beta_prior = bayes_strength*career_games

    #Get posterior distribution, by updating the Gamma distribution with current stats
    alpha_hr_posterior = alpha_hr_prior + home_runs_so_far
    alpha_sb_posterior = alpha_sb_prior + sb_so_far
    beta_posterior = beta_prior + games_played_so_far

    #Sample the posterior distribution for home run and stolen base rates
    lambdas_hr = np.random.gamma(alpha_hr_posterior, 1/beta_posterior, simulations)
    lambdas_sb = np.random.gamma(alpha_sb_posterior, 1/beta_posterior, simulations)

    #Simulate the rest of the season using the home run and stolen base rates using a Poisson distribution
    #Default call of this function generates 1 million simulated seasons per game
    simulated_home_runs = np.random.poisson(lambdas_hr * remaining_games)
    simulated_sb = np.random.poisson(lambdas_sb * remaining_games)
    
    return simulated_home_runs, simulated_sb

#Function to create the individual frames of the GIF
def update_plot(i):
    #Grab the stats after the specified game
    home_runs_so_far = ohtani_stats['Cumulative_HR'].iloc[i]
    sb_so_far = ohtani_stats['Cumulative_SB'].iloc[i]
    games_played_so_far = ohtani_stats['Game_Count'].iloc[i]

    #Simulate the rest of the season and add to true counts
    simulated_home_runs, simulated_sb = simulate_player_performance(home_runs_so_far, sb_so_far, games_played_so_far)
    simulated_home_runs += home_runs_so_far
    simulated_sb += sb_so_far

    #Plot the histogram of the results
    plt.figure(figsize=(8, 8))
    plt.hist2d(simulated_home_runs, simulated_sb, bins=[70, 70], range=[[0, 70], [0, 70]], cmap='Reds', density=True)
    #print(np.max(np.histogram2d(simulated_home_runs, simulated_sb, bins=[70, 70], range=[[0, 70], 10, 70]], density=True)[0]))
    # Calculate probabilities for each club
    prob_30_30 = 100*np.mean((simulated_home_runs >= 30) & (simulated_sb >= 30))
    prob_40_40 = 100*np.mean((simulated_home_runs >= 40) & (simulated_sb >= 40))
    prob_50_50 = 100*np.mean((simulated_home_runs >= 50) & (simulated_sb >= 50))
    
    # Add annotations of the probabilities
    plt.annotate(fr'30-30: {prob_30_30:.0f}%', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, color='black')
    plt.annotate(fr'40-40: {prob_40_40:.0f}%', xy=(0.05, 0.90), xycoords='axes fraction', fontsize=12, color='black')
    plt.annotate(fr'50-50: {prob_50_50:.0f}%', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=12, color='black')

    #Add labels
    plt.xlabel('Home Runs')
    plt.ylabel('Stolen Bases')
    plt.suptitle(f'Ohtani\'s chase for 50-50 - Date: {ohtani_stats["Date"].iloc[i].strftime("%Y-%m-%d")}', x=0.5, y=.95, fontsize=18)
    plt.title('Simulated Full Season Totals', fontsize=10)
    plt.grid(True)
    
    # Save the current frame as an image in the "Frames" directory
    if not os.path.exists('Frames'):
        os.makedirs('Frames')
    plt.savefig(f'Frames/frame_{i:04d}.png')
    if(i==len(ohtani_stats)-1):
        for j in range(50):
          plt.savefig(f'Frames/frame_{(i+j+1):04d}.png')  
    plt.close()

# Load Ohtani's stats
ohtani_stats = pd.read_csv('Ohtani_2024_Game_Log.csv')
ohtani_stats['Date'] = pd.to_datetime(ohtani_stats['Date'])
ohtani_stats['Cumulative_HR'] = ohtani_stats['HR'].cumsum()
ohtani_stats['Cumulative_SB'] = ohtani_stats['SB'].cumsum()
ohtani_stats['Game_Count'] = np.arange(0, len(ohtani_stats))

for i in range(len(ohtani_stats)):
    update_plot(i)

# Directory containing the frames
frames_dir = 'Frames'

# Output GIF file
output_gif = 'ohtani_progression.gif'

# List all the frame files in the directory and sort them alphabetically
frame_files = sorted(
    [f for f in os.listdir(frames_dir) if f.endswith('.png')],
    key=lambda x: int(x.split('_')[1].split('.')[0])
)

# Create a list to hold the images
images = []

# Open each frame and append it to the images list
for frame_file in frame_files:
    frame_path = os.path.join(frames_dir, frame_file)
    img = Image.open(frame_path)
    images.append(img)

# Save the images as a GIF
if images:
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=100,  # Duration of each frame in milliseconds
        loop=0         # Number of loops (0 for infinite)
    )
