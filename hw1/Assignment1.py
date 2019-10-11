# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gamma, poisson
import numpy as np

# library specific settings

sns.set(color_codes=True)
pd.options.display.max_columns = 50

# +
# moving the parent folder to reach helpers
os.chdir('..')

# updating data
from helpers import update_data

# going back to the hw1 folder
os.chdir('hw1')
# -

bets = pd.read_csv("../data/bets.zip")
booking = pd.read_csv("../data/booking.zip")
goals = pd.read_csv("../data/goals.zip")
matches = pd.read_csv("../data/matches.zip")
stats = pd.read_csv("../data/stats.zip")

bets.head()

matches.head()

# ## Data Cleaning

# We need to subset our datasets to English Premiere League matches

# +
#TODO league id's are correct now, no need to this cleaning
# -

PREMIERE_LEAGUE_ID = 148
matches = matches[matches['league_id'] == PREMIERE_LEAGUE_ID].reset_index(drop=True)

# +
# observing unique values for each column to see whether something's wrong

for cols in matches.columns:
  print(cols)
  print(matches[cols].unique())
# -

# We can observe that there are some away teams that are not in English Premiere League (probably international leagues or preparation matches) And there are some matches with NaN scores, and last 5 columns have single observation, they bring no additional information.

# +
# unique home teams

home_teams = (matches[['match_hometeam_id', 'match_hometeam_name']]
              .sort_values('match_hometeam_id')
              .drop_duplicates()
              .reset_index(drop=True))
home_teams.head()
# -

away_teams = (matches[['match_awayteam_id', 'match_awayteam_name']]
              .sort_values('match_awayteam_id')
              .drop_duplicates()
              .reset_index(drop=True))
away_teams.head()

# +
# finding teams that are included in our matches data as both away team and home team

premiere_league_teams = home_teams.merge(away_teams, left_on='match_hometeam_name', right_on='match_awayteam_name')
premiere_league_teams

# +
# observe ManU stored two times as Manchester United and Manchester Utd
# find unique match ids and see how many teams are there

premiere_league_team_ids = premiere_league_teams['match_awayteam_id'].unique().tolist()
print(len(premiere_league_team_ids))
# -

print(len(matches))
matches = matches[(matches['match_awayteam_id'].isin(premiere_league_team_ids)) &
                  (matches['match_hometeam_id'].isin(premiere_league_team_ids))]
print(len(matches))

# Converting epoch column to datetime
matches['timestamp'] = matches['epoch'].apply(lambda x: dt.datetime.fromtimestamp(x))
bets['timestamp'] = bets['odd_epoch'].apply(lambda x: dt.datetime.fromtimestamp(x))

# matches whose match_status is NaN are not played yet, we can remove these data from our base data.
matches[matches['match_hometeam_score'].isnull()].head()

matches = matches.dropna(subset=['match_hometeam_score', 'match_awayteam_score'])

# ## Visualization

# +
sns.distplot(matches.match_hometeam_score, kde=False, bins = 8)
plt.xlabel("Home Goals")
plt.ylabel("Number of Games")
mlest = matches.match_hometeam_score.mean()
k = np.arange(matches.match_hometeam_score.max()+1)

plt.plot(k+0.5, poisson.pmf(k, mlest)*len(matches.match_hometeam_score), 'bo-')

# +
sns.distplot(matches.match_awayteam_score, kde=False, bins = 6)
plt.xlabel("Away Goals")
plt.ylabel("Number of Games")
k = np.arange(matches.match_awayteam_score.max()+1) # observed values for poisson
mean = matches.match_awayteam_score.mean() # mean values of observations

plt.plot(k+0.5,
         poisson.pmf(k, mean)*len(matches.match_awayteam_score),
         'bo-',
         alpha=1)
# -

sns.distplot(matches.match_hometeam_score - matches.match_awayteam_score,
             kde=False, bins = 13, fit=gamma)
plt.xlabel("Home Goals - Away Goals")
plt.ylabel("Number of Games")


print('Mean of home_score - away_score is {}'.format((matches.match_hometeam_score - matches.match_awayteam_score).mean()))
print('Std of home_score - away_score is {}'.format((matches.match_hometeam_score - matches.match_awayteam_score).std()))

# ## 2)
#

print('Number of columns of bets data is {}'.format(len(bets)))
print('Number of columns of bets data whose bettype' +
      'starts with odd (like odd_1 or odd_x2) is {}'
      .format(len(bets[bets['variable'].str.startswith('odd')])))
print('Number of columns of bets data whose bettype ' +
      'starts with odd and of length 5 (like odd_1 or odd_2 or odd_x) is {}'
      .format(len(bets[(bets['variable'].str.startswith('odd')) & (bets['variable'].str.len() == 5)])))

bets = bets[(bets['variable'].str.startswith('odd')) & (bets['variable'].str.len() == 5)]

bets.head()

# pivoting bets data to see the changes with time easily and see the odds in a single row for each match - bookmaker - timestamp
bets_pivoted = bets.pivot_table(index=['match_id', 'odd_bookmakers', 'timestamp'],
                        columns='variable',
                        values='value').reset_index()
bets_pivoted.head()

print(len(bets))
print(len(bets_pivoted) * 3)

# we need to remove these illogical observations
bets_pivoted[(bets_pivoted['odd_1'] <= 1) |
             (bets_pivoted['odd_x'] <= 1) |
             (bets_pivoted['odd_2'] <= 1)]

bets_pivoted = bets_pivoted[(bets_pivoted['odd_1'] > 1) &
                            (bets_pivoted['odd_x'] > 1) &
                            (bets_pivoted['odd_2'] > 1)]

# Since bets are changing by time, I will use final odds announced by bookmakers
# by assuming they are correcting their odds somehow
final_bets = bets_pivoted.groupby(['match_id', 'odd_bookmakers'], as_index=False).last()
final_bets.head()

for cols in ['odd_1', 'odd_x', 'odd_2']:
  final_bets['prob_'+cols] = 1 / final_bets[cols]
final_bets['total'] = final_bets['prob_odd_1'] + final_bets['prob_odd_x'] + final_bets['prob_odd_2']
for cols in ['odd_1', 'odd_x', 'odd_2']:
  final_bets['norm_prob_'+cols] = final_bets['prob_'+cols] / final_bets['total']


matches['result'] = np.where(matches.match_hometeam_score > matches.match_awayteam_score, 1, 0)
matches['result'] = np.where(matches.match_hometeam_score < matches.match_awayteam_score, 2, matches.result)

final_bets = final_bets.merge(matches[['match_id', 'result']], on='match_id')

# +

bins = 10
slices = np.linspace(-1, 1, bins+1, True).astype(np.float)
slices

# -

# TODO: do not assign on the lower limits
def convert_to_bin(num):
  return min([round(slices[i-1], 2) for i,x in enumerate(slices) if num <= x])



final_bets = final_bets[final_bets['odd_bookmakers'].isin(['10Bet', '188BET', '1xBet', 'Betsson'])]
final_bets['diff'] = final_bets['norm_prob_odd_1'] - final_bets['norm_prob_odd_2']
final_bets['bins'] = final_bets['diff'].apply(lambda x: convert_to_bin(x))


match_bins = final_bets.groupby(['bins'], as_index=False)[['match_id']].count()
match_bins

draws = final_bets[final_bets['result'] == 0].groupby(['bins'], as_index=False)[['match_id']].count()
draws

match_bins = match_bins.merge(draws, on='bins', how='outer').fillna(0)
match_bins['draw_ratio'] = match_bins['match_id_y'] / match_bins['match_id_x']
match_bins

ax = sns.scatterplot(x="diff", y="norm_prob_odd_x",
                     sizes=(10, 200),
                     data=final_bets)
plt.plot(match_bins.bins,
         match_bins.draw_ratio,
         'ro-',
         alpha=1)

# '10Bet', '188BET', '1xBet', 'Betsson'
ax = sns.scatterplot(x="diff", y="norm_prob_odd_x",
                     sizes=(10, 200),
                     data=final_bets[final_bets['odd_bookmakers'] == '188BET'])
plt.plot(match_bins.bins,
         match_bins.draw_ratio,
         'ro-',
         alpha=1)

# '10Bet', '188BET', '1xBet', 'Betsson'
ax = sns.scatterplot(x="diff", y="norm_prob_odd_x",
                     sizes=(10, 200),
                     data=final_bets[final_bets['odd_bookmakers'] == '1xBet'])
plt.plot(match_bins.bins,
         match_bins.draw_ratio,
         'ro-',
         alpha=1)

avg_result = final_bets.groupby('bins', as_index=False)[['norm_prob_odd_x']].mean()

# '10Bet', '188BET', '1xBet', 'Betsson'
ax = sns.scatterplot(x="diff", y="norm_prob_odd_x",
                     sizes=(10, 200),
                     data=final_bets[final_bets['odd_bookmakers'] == '188BET'])
plt.plot(match_bins.bins,
         match_bins.draw_ratio,
         'ro-',
         alpha=1)
plt.plot(avg_result.bins,
         avg_result.norm_prob_odd_x,
         'go-',
         alpha=1)




