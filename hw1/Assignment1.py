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

import os
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
import numpy as np


## library options
sns.set(color_codes=True, palette="colorblind")
sns.set(rc={'figure.figsize':(12,8)})
pd.options.display.max_columns = 50

# +
# Reading the data

bets = pd.read_csv("../data/bets.zip")
booking = pd.read_csv("../data/booking.zip")
goals = pd.read_csv("../data/goals.zip")
matches = pd.read_csv("../data/matches.zip")
stats = pd.read_csv("../data/stats.zip")
# -

bets.head(1)

booking.head(1)

goals.head(1)

matches.head(1)

stats.head(1)

# # Exploratory Data Analysis & Data Cleaning
# We need to subset our datasets to English Premiere League matches

print(len(bets))
print(len(booking))
print(len(goals))
print(len(matches))
print(len(stats))

PREMIERE_LEAGUE_ID = 148
matches = matches[matches['league_id'] == PREMIERE_LEAGUE_ID].reset_index(drop=True)
bets = bets.merge(matches[['match_id']], on='match_id').reset_index(drop=True)
booking = booking.merge(matches[['match_id']], on='match_id').reset_index(drop=True)
goals = goals.merge(matches[['match_id']], on='match_id').reset_index(drop=True)
stats = stats.merge(matches[['match_id']], on='match_id').reset_index(drop=True)

print(len(bets))
print(len(booking))
print(len(goals))
print(len(matches))
print(len(stats))

# +
# observing unique values for each column to see whether something's wrong

for cols in matches.columns:
  if cols not in ['match_id', 'epoch']:
    print(cols)
    print(matches[cols].unique())
# -

# - There are some matches with Finished and nan status.
# - Names of teams can be stored differently (Manchester United and Manchester Utd or West Ham and West Ham (Eng))
# - There were some away teams that are not from Premiere league (champions league matches), but they're removed somehow, just beware of this.
# - There are nan scores in some of the scores columns, these should be removed.

# Converting epoch column to datetime
matches['timestamp'] = matches['epoch'].apply(lambda x: dt.datetime.fromtimestamp(x))
bets['timestamp'] = bets['odd_epoch'].apply(lambda x: dt.datetime.fromtimestamp(x))

# matches whose match_status is NaN are not played yet, we can remove these data from our base data.
print('Number of rows before removing NaN\'s: {}'.format(len(matches)))
matches = matches.dropna(subset=['match_status', 'match_hometeam_score', 'match_awayteam_score'])
print('Number of rows after removing NaN\'s: {}'.format(len(matches)))

# ## Task1

# +
# Drawing histogram of home goals

# range of observations
bin_number = int(matches.match_hometeam_score.max() - 
                 matches.match_hometeam_score.min())

sns.distplot(matches.match_hometeam_score, 
             kde=False, 
             bins=bin_number)

plt.xlabel("Home Goals")
plt.ylabel("Number of Games")

# Fitting Poisson distribution on it, by using lambda as mean of all observations
mean = matches.match_hometeam_score.mean()
k = np.arange(matches.match_hometeam_score.max()+1)

plt.plot(k+0.5, poisson.pmf(k, mean)*len(matches.match_hometeam_score), 'o-')

# +
# Drawing histogram of away goals

bin_number = int(matches.match_awayteam_score.max() - 
                 matches.match_awayteam_score.min())

sns.distplot(matches.match_awayteam_score, 
             kde=False, 
             bins=bin_number)

plt.xlabel("Away Goals")
plt.ylabel("Number of Games")

# Fitting Poisson on to it
k = np.arange(matches.match_awayteam_score.max()+1) 
# length of observed values to calculate prob for each
# like [0, 1, 2, 3, ..., n], values will be shifted 0.5 to be centered
mean = matches.match_awayteam_score.mean() # mean values of observations

plt.plot(k+0.5, poisson.pmf(k, mean)*len(matches.match_awayteam_score), 'o-')

# +
bin_number = int((matches.match_hometeam_score - matches.match_awayteam_score).max() - 
                 (matches.match_hometeam_score - matches.match_awayteam_score).min())

sns.distplot(matches.match_hometeam_score - matches.match_awayteam_score,
             kde=False,
             bins=13, 
             fit=norm) # this will fit normal distribution on it
plt.xlabel("Home Goals - Away Goals")
plt.ylabel("Number of Games")
# -

print('Mean of home_score - away_score is {}'
      .format((matches.match_hometeam_score - matches.match_awayteam_score).mean()))
print('Std of home_score - away_score is {}'
      .format((matches.match_hometeam_score - matches.match_awayteam_score).std()))

# Poisson distribution fits well to away and home scores case, whereas the difference between two teams' goals is distributed normally with mean 0.37 and standart deviation 1.89

# ## Task2
#

# subsetting bets to odd1 oddx odd2 only
# odd values should be more than 
print(len(bets))
bets = bets[bets['value'] > 1]
print(len(bets))
bets = bets[bets['variable'].isin(['odd_1', 'odd_x', 'odd_2'])]
print(len(bets))

# +
# pivoting bets data to see the changes with time easily and 
# see the odds in a single row for each match - bookmaker - timestamp

bets = bets.pivot_table(index=['match_id', 'odd_bookmakers', 'timestamp'],
                        columns='variable',
                        values='value').reset_index()

# reordering columns
bets = bets[['match_id', 'odd_bookmakers', 'odd_1', 'odd_x', 'odd_2', 'timestamp']]
# -

bets.head()

# +
# Since bets are changing by time, I will use final odds announced by bookmakers
# by assuming they are correcting their odds somehow

final_bets = bets.groupby(['match_id', 'odd_bookmakers'], as_index=False).last()
print(len(bets))
print(len(final_bets))
# -

final_bets.head()

# +
# Calculating implied naive probabilities and creating new prob_odd_1(x,2) columns
for cols in ['odd_1', 'odd_x', 'odd_2']:
  final_bets['prob_'+cols] = 1 / final_bets[cols]

# Summing all naive probabilities for each bookmaker & match (this will give us 1 + margin of bookmaker)
final_bets['total'] = final_bets['prob_odd_1'] + final_bets['prob_odd_x'] + final_bets['prob_odd_2']

# normalizin odd by removing margin share from each of them
for cols in ['odd_1', 'odd_x', 'odd_2']:
  final_bets['norm_prob_'+cols] = final_bets['prob_'+cols] / final_bets['total']


# +
# creates a result column 1, 0 or 2 for home win, draw, away win accordingly

matches['result'] = np.where(matches.match_hometeam_score > matches.match_awayteam_score, 
                             1, 0)
# if away > home, then returns 2. otherwise returns the previous result value 
# (which is 1 if home > away and 0 otherwise)

matches['result'] = np.where(matches.match_hometeam_score < matches.match_awayteam_score, 
                             2, matches.result)

# joining result info into the final bets table

final_bets = final_bets.merge(matches[['match_id', 'result']], 
                              on='match_id')
final_bets.head(1)

# +
# to create an array like [-1. , -0.8, -0.6, -0.4, -0.2,  0. ,  0.2,  0.4,  0.6,  0.8,  1. ]
bins = 10
slices = np.linspace(-1, 1, bins+1, True).astype(np.float)

def convert_to_bin(num):
  """
  for each num returns the average value of previous and next observations in slices array
  
  ex:
  
  array = [-1. , -0.8, -0.6, -0.4, -0.2,  0. ,  0.2,  0.4,  0.6,  0.8,  1. ]
  num = 0.72
  
  output = (0.6 + 0.8) / 2
  """
  return min([round((slices[i-1] + slices[i]) / 2, 2) for i,x in enumerate(slices) if num <= x])


# -

final_bets['diff'] = final_bets['norm_prob_odd_1'] - final_bets['norm_prob_odd_2']
final_bets['bins'] = final_bets['diff'].apply(lambda x: convert_to_bin(x))


def plot_bookmaker_odds(df, plot_name, critical_df=None):
  prob_bins = df.groupby(['bins'], as_index=False)[['match_id']].count()
  draws = df[df['result'] == 0].groupby(['bins'], as_index=False)[['match_id']].count()
  prob_bins = prob_bins.merge(draws, on='bins', how='outer').fillna(0)
  prob_bins['draw_ratio'] = prob_bins['match_id_y'] / prob_bins['match_id_x']

  avg_result = df.groupby('bins', as_index=False)[['norm_prob_odd_x']].mean()
  prob_bins = prob_bins.merge(avg_result, on='bins', how='outer').fillna(0)
  prob_bins.columns = ['bins', 'total_matches', 'draw_matches', 'draw_ratio', 'mean_draw_prob']
  ax = sns.scatterplot(x="diff", 
                       y="norm_prob_odd_x",
                       data=df).set_title(plot_name, fontsize = 15)
  plt.plot(prob_bins.bins,
           prob_bins.draw_ratio,
           'ko-',
           alpha=1)
  plt.plot(prob_bins.bins,
           prob_bins.mean_draw_prob,
           'ro-',
           alpha=1)
  label_list = ['Actual Draw Ratios', 'Mean Draw Probs', 'Implied Draw Probs']
  if critical_df is not None:
    critical_prob_bins = critical_df.groupby(['bins'], as_index=False)[['match_id']].count()
    critical_draws = critical_df[critical_df['result'] == 0].groupby(['bins'], as_index=False)[['match_id']].count()
    critical_prob_bins = critical_prob_bins.merge(critical_draws, on='bins', how='outer').fillna(0)
    critical_prob_bins['draw_ratio'] = critical_prob_bins['match_id_y'] / critical_prob_bins['match_id_x']
    # cric_avg_result = critical_df.groupby('bins', as_index=False)[['norm_prob_odd_x']].mean()
    # critical_prob_bins = critical_prob_bins.merge(cric_avg_result, on='bins', how='outer').fillna(0)
    # critical_prob_bins.columns = ['bins', 'total_matches', 'draw_matches', 'draw_ratio', 'mean_draw_prob']
    plt.plot(critical_prob_bins.bins,
             critical_prob_bins.draw_ratio,
             'co-',
             alpha=1)
    # plt.plot(critical_prob_bins.bins,
    #          critical_prob_bins.mean_draw_prob,
    #          'o-',
    #          alpha=1)
  
    label_list = ['Actual Draw Ratios', 'Mean Draw Probs', 'Actual Draw Ratios - Cleaned', 'Implied Draw Probs']

  plt.xlabel('P(Home) - P(Away)')
  plt.ylabel('P(Draw)')
  plt.legend(loc='upper right', labels=label_list)


plot_bookmaker_odds(final_bets , 'All Bookmakers')

plot_bookmaker_odds(final_bets[final_bets['odd_bookmakers'] == 'Betsson'], 'Betsson')

plot_bookmaker_odds(final_bets[final_bets['odd_bookmakers'] == '10Bet'], '10Bet')

plot_bookmaker_odds(final_bets[final_bets['odd_bookmakers'] == '188BET'], '188BET')

plot_bookmaker_odds(final_bets[final_bets['odd_bookmakers'] == '1xBet'], '1xBet')

# Separating into 20 bins has very ups and downs, so it's hard to conclude. Therefore I decided to use 10-bins, separated as [-1, -0.8, -0.6, ..., 0.4, 0.6, 0.8, 1] It can be concluded that we can observe significant difference between actual draw probability and implied draw probability when home and away teams are similar (ther probability of winning is near zero)
#
# Since there is not that much difference when we look at the all bookmakers case, we can have a feeling that some bookmakers are predicting better than what I randomly selected (4 bookmakers above), so bookmakers are not predicting outcomes of the matches equally.

# ## Task3

# +
# since there are some values like 90+2, time column is stored as string
# so in order to sort correctly, we need to convert 4 to 04 (adding one zero to the left)
goals['time'] = goals['time'].apply(lambda x: x.zfill(2))
booking['time'] = booking['time'].apply(lambda x: x.zfill(2))

# since we will deal with changes in scores, all observations must be time ordered
goals = goals.sort_values(['match_id', 'time'])


# -

def score_difference(col_str):
  """
  first converts '4 - 1' to [4, 1]
  and then returns max(score_list) - min(score_list), 3
  """
  score_list = [int(i) for i in col_str.split('-')]
  return max(score_list) - min(score_list)


# find all score differences for each match in each goal (we will use this info 
# to check whether this difference increased with last goal or not)
goals['score_diff'] = goals['score'].apply(lambda x: score_difference(x))
goals["prev_score"] = goals["score_diff"].groupby(goals['match_id']).shift(1)

goals.head(10)

extra_time_goals = goals[goals['time'] > '90']
critical_goals = extra_time_goals[(extra_time_goals['score_diff'] == 0) | (extra_time_goals['prev_score'] == 0)]

critical_goals.head()

print(len(extra_time_goals))
print(len(critical_goals))

red_card_matches = booking[(booking['time'] < '30') & (booking['card'] == 'red card')]

print(len(red_card_matches))

critical_matches = critical_goals['match_id'].to_list()
red_card_matches = red_card_matches['match_id'].to_list()
matches_to_remove = critical_matches + red_card_matches
print(len(matches_to_remove))

critical_df = final_bets[(~final_bets['match_id'].isin(matches_to_remove))]
plot_bookmaker_odds(final_bets, 
                    'All Bookmakers Critical Matches Removed', 
                    critical_df)

critical_df = final_bets[(~final_bets['match_id'].isin(matches_to_remove)) & 
                         (final_bets['odd_bookmakers'] == '1xBet')]
plot_bookmaker_odds(final_bets[final_bets['odd_bookmakers'] == '1xBet'], 
                    '1xBet Critical Matches Removed', 
                    critical_df)

critical_df = final_bets[(~final_bets['match_id'].isin(matches_to_remove)) & 
                         (final_bets['odd_bookmakers'] == 'Betsson')]
plot_bookmaker_odds(final_bets[final_bets['odd_bookmakers'] == 'Betsson'], 
                    'Betsson Critical Matches Removed', 
                    critical_df)

critical_df = final_bets[(~final_bets['match_id'].isin(matches_to_remove)) & 
                         (final_bets['odd_bookmakers'] == '10Bet')]
plot_bookmaker_odds(final_bets[final_bets['odd_bookmakers'] == '10Bet'], '10Bet Critical Matches Removed', critical_df)

critical_df = final_bets[(~final_bets['match_id'].isin(matches_to_remove)) & 
                         (final_bets['odd_bookmakers'] == '188BET')]
plot_bookmaker_odds(final_bets[final_bets['odd_bookmakers'] == '188BET'], '188BET Critical Matches Removed', critical_df)

# When we remove extra time goals from our dataset, observed ratios of draws tend to decrease. This might be explained by when there is one goal difference between teams, losing teams might be attacking to gain 1 point, so matches with extra time goals has more probability of draw than normal matches. But there are still place to get money whether there exists extra goal or not. Early red cards does not occur that much, so it's hard to conclude on them.
