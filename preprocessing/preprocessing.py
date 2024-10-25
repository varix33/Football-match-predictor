import numpy as np
import pandas as pd
from tqdm import tqdm
import os

"""
-----------------------------------------------------------------------------------------------------------------
| This script is an optimised version of the original, it isn't representative of the way we dealt
| with the subject, and so, doesn't contain food for thought and analysis with graphics, statistical data.
| To have an overview of this part, you can refer to the report and the file annexe.py
-----------------------------------------------------------------------------------------------------------------
"""

# ---- Sets ----------------------------------------------------------------------

# a set of countries to rename
replacementsRankingName = {
    'St. Vincent / Grenadines': 'Saint Vincent and the Grenadines',
    'St. Vincent and the Grenadines': 'Saint Vincent and the Grenadines',
    'St Vincent and the Grenadines': 'Saint Vincent and the Grenadines',
    'Cape Verde Islands': 'Cape Verde',
    'Cabo Verde': 'Cape Verde',
    'Korea Republic': 'South Korea',
    'Korea DPR': 'North Korea',
    'USA': 'United States',
    'United Kingdom': 'England',
    'IR Iran': 'Iran',
    'St. Kitts and Nevis': 'Saint Kitts and Nevis',
    'St Kitts and Nevis': 'Saint Kitts and Nevis',
    'St. Lucia': 'Saint Lucia',
    'St Lucia': 'Saint Lucia',
    'Brunei Darussalam': 'Brunei',
    'Congo DR': 'DR Congo',
    'Zaire': 'DR Congo',
    'Türkiye': 'Turkey',
    'Curacao': 'Curaçao',
    'Côte d\'Ivoire': 'Ivory Coast',
    'Sao Tome e Principe': 'São Tomé and Príncipe',
    'São Tomé e Príncipe': 'São Tomé and Príncipe',
    'US Virgin Islands': 'United States Virgin Islands',
    'Aotearoa New Zealand': 'New Zealand',
    'The Gambia': 'Gambia',
    'FYR Macedonia': 'North Macedonia',
    'Swaziland': 'Eswatini',
    'Czechia': 'Czech Republic',
    'Kyrgyz Republic': 'Kyrgyzstan',
    'Hong Kong, China': 'Hong Kong',
    'Chinese Taipei': 'Taiwan',
}

# a set of countries who are members of the FIFA in 2023
fifa_member_countries = {
    'Argentina', 'France', 'Brazil', 'England', 'Belgium', 'Portugal', 'Netherlands', 'Spain', 'Italy', 'Croatia',
    'United States', 'Mexico', 'Morocco', 'Switzerland', 'Uruguay', 'Germany', 'Colombia', 'Japan', 'Denmark',
    'Senegal',
    'Iran', 'Ukraine', 'Sweden', 'South Korea', 'Austria', 'Peru', 'Australia', 'Wales', 'Serbia', 'Hungary',
    'Poland', 'Tunisia', 'Algeria', 'Scotland', 'Egypt', 'Ecuador', 'Chile', 'Turkey', 'Russia', 'Nigeria',
    'Czech Republic',
    'Norway', 'Cameroon', 'Panama', 'Canada', 'Costa Rica', 'Mali', 'Romania', 'Venezuela', 'Slovakia', 'Greece',
    'Ivory Coast', 'Paraguay', 'Slovenia', 'Jamaica', 'Burkina Faso', 'Saudi Arabia', 'Republic of Ireland', 'Albania',
    'Ghana', 'Qatar', 'Finland', 'Bosnia and Herzegovina', 'South Africa', 'DR Congo', 'North Macedonia', 'Iceland',
    'Iraq',
    'United Arab Emirates', 'Montenegro', 'Israel', 'Oman', 'Uzbekistan', 'Cape Verde', 'Northern Ireland', 'Georgia',
    'El Salvador',
    'Honduras', 'China PR', 'Guinea', 'Zambia', 'Jordan', 'Bahrain', 'Bulgaria', 'Bolivia', 'Gabon', 'Luxembourg',
    'Haiti', 'Curaçao', 'Uganda', 'Equatorial Guinea', 'Syria', 'Benin', 'Vietnam', 'Armenia', 'Palestine',
    'Kyrgyzstan', 'Kazakhstan', 'Trinidad and Tobago', 'Belarus', 'Mauritania', 'India', 'New Zealand', 'Lebanon',
    'Kosovo', 'Congo', 'Guatemala', 'Madagascar', 'Tajikistan', 'Kenya', 'Guinea-Bissau', 'Thailand', 'Mozambique',
    'Namibia', 'North Korea', 'Angola', 'Gambia', 'Estonia', 'Togo', 'Azerbaijan', 'Tanzania', 'Sierra Leone', 'Malawi',
    'Cyprus', 'Zimbabwe', 'Libya', 'Central African Republic', 'Comoros', 'Niger', 'Sudan', 'Solomon Islands', 'Latvia',
    'Nicaragua', 'Lithuania', 'Faroe Islands', 'Kuwait', 'Malaysia', 'Philippines', 'Antigua and Barbuda', 'Rwanda',
    'Turkmenistan', 'Burundi', 'Ethiopia', 'Suriname', 'Indonesia', 'Eswatini', 'Saint Kitts and Nevis', 'Botswana',
    'Dominican Republic', 'Hong Kong', 'Liberia', 'Taiwan', 'Lesotho', 'Afghanistan', 'Singapore', 'Yemen',
    'Moldova', 'Myanmar', 'Andorra', 'Guyana', 'Maldives', 'New Caledonia', 'Tahiti', 'Puerto Rico', 'Papua New Guinea',
    'Saint Lucia', 'South Sudan', 'Vanuatu', 'Cuba', 'Fiji', 'Malta', 'Bermuda', 'Nepal', 'Barbados',
    'Saint Vincent and the Grenadines', 'Grenada', 'Mauritius', 'Cambodia', 'Chad', 'Montserrat', 'Belize', 'Bhutan',
    'Bangladesh',
    'Dominica', 'Cook Islands', 'São Tomé and Príncipe', 'Macau', 'Laos', 'Djibouti', 'Mongolia', 'Brunei', 'Aruba',
    'Pakistan', 'Cayman Islands', 'Seychelles', 'Somalia', 'Timor-Leste', 'Gibraltar', 'Bahamas', 'Liechtenstein',
    'Turks and Caicos Islands', 'Sri Lanka', 'Guam', 'British Virgin Islands', 'United States Virgin Islands',
    'Anguilla', 'San Marino'
}

# a list containing 4 tournament group :
#   group 4 -> Friendly and Lesser-Known Tournaments
#   group 3 -> Continental Tournaments and Secondary Qualifications
#   group 2 -> Continental Tournaments and Major Qualifications
#   group 1 -> Major Tournaments
tournament_classification = [

    # Group 4 - Friendly and Lesser-Known Tournaments
    {'Friendly', 'Lunar New Year Cup', 'Dynasty Cup', "King's Cup", 'Nehru Cup', 'Copa Paz del Chaco', 'Baltic Cup',
     'Kirin Cup', 'Korea Cup', 'USA Cup', 'Merdeka Tournament', 'South Pacific Games', 'Simba Tournament',
     'Gold Cup qualification', 'SKN Football Festival', 'Arab Cup', 'UNIFFAC Cup', 'Nordic Championship',
     'Millennium Cup', 'Cup of Ancient Civilizations', "Prime Minister's Cup", 'The Other Final', 'EAFF Championship',
     'TIFOCO Tournament', 'Afro-Asian Games', 'Copa del Pacífico', 'AFC Challenge Cup qualification', 'VFF Cup',
     'Dragon Cup', 'Nile Basin Tournament', 'Nations Cup', 'Copa Confraternidad', 'Pacific Mini Games',
     'Intercontinental Cup', 'AFF Championship qualification', 'Jordan International Tournament',
     'Tri Nation Tournament',
     'CAFA Nations Cup', 'Mauritius Four Nations Cup',
     'ABCS Tournament', 'African Cup of Nations qualification', 'African Nations Championship',
     'Amílcar Cabral Cup', 'CFU Caribbean Cup qualification', 'CONMEBOL–UEFA Cup of Champions', 'Confederations Cup',
     'Cyprus International Tournament', 'Dunhill Cup', 'Gold Cup', 'King Hassan II Tournament', 'Kirin Challenge Cup',
     "MSG Prime Minister's Cup", 'Mahinda Rajapaksa Cup', 'Malta International Tournament', 'Navruz Cup', 'OSN Cup',
     'Oceania Nations Cup', 'Superclásico de las Américas', 'Three Nations Cup', 'Tournoi de France',
     'United Arab Emirates Friendship Tournament', 'Windward Islands Tournament'},

    # Group 3 - Continental Tournaments and Secondary Qualifications
    {'CFU Caribbean Cup', 'CECAFA Cup', 'COSAFA Cup', 'AFF Championship', 'Gulf Cup', 'Caribbean Cup qualification',
     'COSAFA Cup qualification', 'WAFF Championship', 'AFC Challenge Cup', 'SAFF Cup', 'UNCAF Cup', 'Pacific Games'},

    # Group 2 - Continental Tournaments and Major Qualifications
    {'UEFA Nations League', 'CONCACAF Nations League', 'Copa América qualification', 'UEFA Euro qualification',
     'AFC Asian Cup qualification', 'African Nations Championship qualification', 'FIFA World Cup qualification',
     'CONCACAF Nations League qualification', 'Oceania Nations Cup qualification', 'Arab Cup qualification'},

    # Group 1 - Major Tournaments
    {'FIFA World Cup', 'UEFA Euro', 'Copa América', 'AFC Asian Cup', 'African Cup of Nations', 'CONCACAF Gold Cup'}
]


# ---- sub functions ----------------------------------------------------------------

"""
/ return the target class of a data(match)
/ 0 if home_score win, 1 if away_score win
"""
def who_win(match):
    if match['home_score'] > match['away_score']:
        return 0
    elif match['home_score'] < match['away_score']:
        return 1
    # if draw : the winner is the team which have the best rank
    else:
        if match['rank_home'] < match['rank_away']:
            return 0
        else:
            return 1


"""
/ return the features for a match :
/   win_score_5last_enc_home -> a score for home_team calculated on the victory against away_team on the 5 last previous matches
/   win_score_5last_enc_away -> a score for away_team calculated on the victory against home_team on the 5 last previous matches
/   this score is better when the team win a more recent match.
/ param base : the dataset which contains all the match
/ param match : the match
"""
def get_win_score_5last_enc(base, match):

    win_score_5last_enc_home = 0
    win_score_5last_enc_away = 0

    # get all the last matches between those 2 teams
    encounters = base[(base['teams'] == match['teams']) & (base['date'] < match['date'])]

    if not encounters.empty:
        last_five_enc = encounters.tail(5)  # the last five matches

        i = 5  # a decrementer to give more points on the more recent victory
        for _, match in last_five_enc.iterrows():
            # if home_team in "match" won
            if match['winner'] == 0:
                if match['home_team'] == match['home_team']:
                    win_score_5last_enc_home += 1 * i
                else:
                    win_score_5last_enc_away += 1 * i
            # if away_team in "match" won
            else:
                if match['away_team'] == match['away_team']:
                    win_score_5last_enc_away += 1 * i
                else:
                    win_score_5last_enc_home += 1 * i
            i = i - 1
    return win_score_5last_enc_home, win_score_5last_enc_away

"""
/ return the dataset with two new features :
/   win_score_5last_enc_home -> a score for home_team calculated on the victory against away_team on the 5 last previous matches
/   win_score_5last_enc_away -> a score for away_team calculated on the victory against home_team on the 5 last previous matches
/   this score is better when the team win a more recent match.
/ param base : the dataset which contains all the match
/ param target : the dataset to return
"""
def add_features_win_score_5last_enc(target, base):

    # we don't want to modify the original datasets
    target = target.copy(deep=True)
    base = base.copy(deep=True)

    # type conversion in order to do operation between those date
    target['date'] = pd.to_datetime(target['date'])
    base['date'] = pd.to_datetime(base['date'])

    # create a temporal column which contains a set of both country names
    target['teams'] = target.apply(lambda row: frozenset([row['home_team'], row['away_team']]), axis=1)
    base['teams'] = base.apply(lambda row: frozenset([row['home_team'], row['away_team']]), axis=1)

    # create the 2 new features
    target[['win_score_5last_enc_home', 'win_score_5last_enc_away']] = 0

    # use a progression bar with tqdm
    total_iterations = len(target)
    tqdm.pandas(total=total_iterations)

    for _, tMatch in tqdm(target.iterrows(), total=total_iterations, desc="Adding features (1/4)"):
        win_score_5last_enc_home, win_score_5last_enc_away = get_win_score_5last_enc(base, tMatch)
        target.at[tMatch.name, 'win_score_5last_enc_home'] = win_score_5last_enc_home
        target.at[tMatch.name, 'win_score_5last_enc_away'] = win_score_5last_enc_away

    target.drop(columns={'teams'}, inplace=True)  # drop the temporal column
    return target


"""
/ return the feature for a match :
/   diff_goal_last_enc -> the difference of goals scored by both teams during their last match
/ param base : the dataset which contains all the match
/ param match : the match
"""
def get_diff_goals_last_enc(base, match):
    # get all the last matches between those 2 teams
    encounters = base[(base['teams'] == match['teams']) & (base['date'] < match['date'])]

    if not encounters.empty:
        last_enc = encounters.iloc[-1]  # the last match

        if match['home_team'] == last_enc['home_team']:
            return last_enc['home_score'] - last_enc['away_score']
        else:
           return last_enc['away_score'] - last_enc['home_score']
    return 0


"""
/ return the dataset with two new features :
/   diff_goal_last_enc -> the difference of goals scored by both teams during their last match
/ param base : the dataset which contains all the match
/ param target : the dataset to return
"""
def add_feature_diff_goals_last_enc(target, base):

    # we don't want to modify the original datasets
    target = target.copy(deep=True)
    base = base.copy(deep=True)

    # type conversion in order to do operation between those date
    target['date'] = pd.to_datetime(target['date'])
    base['date'] = pd.to_datetime(base['date'])

    # create a temporal column which contains a set of both country names
    target['teams'] = target.apply(lambda row: frozenset([row['home_team'], row['away_team']]), axis=1)
    base['teams'] = base.apply(lambda row: frozenset([row['home_team'], row['away_team']]), axis=1)

    # create the new features
    target['diff_goals_last_enc'] = 0

    # use a progression bar with tqdm
    total_iterations = len(target)
    tqdm.pandas(total=total_iterations)

    for _, tMatch in tqdm(target.iterrows(), total=total_iterations, desc="Adding features (2/4)"):
        target.at[tMatch.name, 'diff_goals_last_enc'] = get_diff_goals_last_enc(base, tMatch)

    target.drop(columns={'teams'}, inplace=True)  # drop the temporal column
    return target

"""
/ return the features for a match :
/   avg_goals_last_five_home -> average of goals scored by home_team during their 5 last matches
/   avg_goals_last_five_away -> average of goals scored by away_team during their 5 last matches
/ param base : the dataset which contains all the match
/ param match : the match
"""
def get_avg_goal_last_5matches(base, match):

    avg_goals_last_five_home = 0
    avg_goals_last_five_away = 0

    # get the 5 last matches of home_team
    last_home_team_matches = base[((base['home_team'] == match['home_team']) | (base['away_team'] == match['home_team'])) & (base['date'] < match['date'])].tail(5)
    # get the 5 last matches of away_team
    last_away_team_matches = base[((base['home_team'] == match['away_team']) | (base['away_team'] == match['away_team'])) & (base['date'] < match['date'])].tail(5)

    homeGoal = []  # list of goals scored by home_team
    for _, m in last_home_team_matches.iterrows():
        if m['home_team'] == match['home_team']:
            homeGoal.append(m['home_score'])
        else:
            homeGoal.append(m['away_score'])

    # set the feature of home_team
    if homeGoal:
        avg_goals_last_five_home = np.mean(homeGoal)

    awayGoal = []  # list of goals scored by away_team
    for _, m in last_away_team_matches.iterrows():
        if m['home_team'] == match['away_team']:
            awayGoal.append(m['home_score'])
        else:
            awayGoal.append(m['away_score'])

    # set the feature of home_team
    if awayGoal:
        avg_goals_last_five_away = np.mean(awayGoal)

    return avg_goals_last_five_home, avg_goals_last_five_away


""" 
/ return the dataset with two new features :
/   avg_goals_last_five_home -> average of goals scored by home_team during their 5 last matches
/   avg_goals_last_five_away -> average of goals scored by away_team during their 5 last matches
/ param base : the dataset which contains all the match
/ param target : the dataset to return
"""
def add_features_avg_goal_last_5matches(target, base):

    # we don't want to modify the original datasets
    target = target.copy(deep=True)
    base = base.copy(deep=True)

    # type conversion in order to do operation between those date
    target['date'] = pd.to_datetime(target['date'])
    base['date'] = pd.to_datetime(base['date'])

    # create the 2 new features
    target[['avg_goals_last_five_home', 'avg_goals_last_five_away']] = 0.0

    # use a progression bar with tqdm
    total_iterations = len(target)
    tqdm.pandas(total=total_iterations)

    for _, tMatch in tqdm(target.iterrows(), total=total_iterations, desc="Adding features (3/4)"):
        avg_goals_last_five_home, avg_goals_last_five_away = get_avg_goal_last_5matches(base, tMatch)
        target.at[tMatch.name, 'avg_goals_last_five_home'] = avg_goals_last_five_home
        target.at[tMatch.name, 'avg_goals_last_five_away'] = avg_goals_last_five_away

    return target


"""
/ return the features for a match :
/   win_rate_tournament_home -> the win rate of home_team for this type of tournament on their five last matches 
/   win_rate_tournament_away -> the win rate of away_team for this type of tournament on their five last matches
/ param base : the dataset which contains all the match
/ param match : the match
"""
def get_win_rate_tournament(base, match):

    win_rate_tournament_home = 0
    win_rate_tournament_away = 0

    # get the 5 last matches of home_team in this type of tournament
    last_home_team_matches = base[((base['home_team'] == match['home_team']) | (base['away_team'] == match['home_team'])) & (base['date'] < match['date']) & (base['tournament'] == match['tournament'])].tail(5)
    # get the 5 last matches of away_team in this type of tournament
    last_away_team_matches = base[((base['home_team'] == match['away_team']) | (base['away_team'] == match['away_team'])) & (base['date'] < match['date']) & (base['tournament'] == match['tournament'])].tail(5)

    # calculate the win_rate of home team
    nb_match = len(last_home_team_matches)
    nb_win = 0
    for _, m in last_home_team_matches.iterrows():
        if m['home_team'] == match['home_team']:
            if m['winner'] == 0:
                nb_win += 1
        else:
            if m['winner'] == 1:
                nb_win += 1

    # set the feature of home_team
    if nb_match > 0:
        win_rate_tournament_home = float(nb_win) / nb_match

    # calculate the win_rate of away team
    nb_match = len(last_away_team_matches)
    nb_win = 0
    for _, m in last_away_team_matches.iterrows():
        if m['away_team'] == match['away_team']:
            if m['winner'] == 1:
                nb_win += 1
        else:
            if m['winner'] == 0:
                nb_win += 1

    # set the feature of away_team
    if nb_match > 0:
        win_rate_tournament_away = float(nb_win) / nb_match

    return win_rate_tournament_home, win_rate_tournament_away

""" 
/ return the dataset with two new features:
/   win_rate_tournament_home -> the win rate of home_team for this type of tournament on their five last matches 
/   win_rate_tournament_away -> the win rate of away_team for this type of tournament on their five last matches
/ param base : the dataset which contains all the match
/ param target : the dataset to return
"""
def add_features_win_rate_tournament(target, base):

    # we don't want to modify the original datasets
    target = target.copy(deep=True)
    base = base.copy(deep=True)

    # type conversion in order to do operation between those date
    target['date'] = pd.to_datetime(target['date'])
    base['date'] = pd.to_datetime(base['date'])

    # create the 2 new features
    target[['win_rate_tournament_home', 'win_rate_tournament_away']] = 0.0

    # use a progression bar with tqdm
    total_iterations = len(target)
    tqdm.pandas(total=total_iterations)

    for _, tMatch in tqdm(target.iterrows(), total=total_iterations, desc="Adding features (4/4)"):
        win_rate_tournament_home, win_rate_tournament_away = get_win_rate_tournament(base, tMatch)
        target.at[tMatch.name, 'win_rate_tournament_home'] = win_rate_tournament_home
        target.at[tMatch.name, 'win_rate_tournament_away'] = win_rate_tournament_away

    return target


""" 
/ convert a tournament into an integer which represent the type of tournament
/ param name : the name of the tournament 
"""
def get_rank_tournament(name):
    for i in range(len(tournament_classification)):
        if name in tournament_classification[i]:
            return i


""" 
/ return the dataset target with all the new features added
/ param base : the dataset which contains all the match
/ param target : the dataset to modify
"""
def add_features(target, base):

    # we don't want to modify the original datasets
    target = target.copy(deep=True)

    tqdm.pandas()
    # add 2 new features
    target = add_features_win_score_5last_enc(target, base)
    # add 1 new feature
    target = add_feature_diff_goals_last_enc(target, base)
    # add 2 new features
    target = add_features_avg_goal_last_5matches(target, base)
    # add 2 new features
    target = add_features_win_rate_tournament(target, base)

    return target

""" 
/ return the dataset resampled 
/ param rankingData : the dataset to return
"""
def resampling(rankingData):

    # we don't want to modify the original datasets
    rankingDataResample = rankingData.copy(deep=True)
    rankingDataResample['date'] = pd.to_datetime(rankingData['rank_date'])
    rankingDataResample.set_index('date', inplace=True)
    # resampling of the dataset rankingData
    rankingDataResample = rankingDataResample.groupby('country_full', as_index=False).resample('D').first()
    rankingDataResample.ffill(inplace=True)
    #rankingDataResample.reset_index(drop=True, inplace=True)

    return rankingDataResample


""" 
/ return a merged dataset 
/ param rankingDataResample : the dataset ranking resampled
/ param resultsData :  the dataset results
"""
def merging(rankingDataResample, resultsData):

    # we don't want to modify the original datasets
    resultsDataTemp = resultsData.copy(deep=True)

    resultsDataTemp['date'] = pd.to_datetime(resultsDataTemp['date'])

    # merging both dataset in only one dataset(reraBase) to have the current world ranks of each team in each matches
    reraBase = pd.merge(resultsDataTemp, rankingDataResample, left_on=['date', 'home_team'],right_on=['date', 'country_full'], how='left')
    reraBase = pd.merge(reraBase, rankingDataResample, left_on=['date', 'away_team'], right_on=['date', 'country_full'],how='left')

    # rename the new columns after the merge
    reraBase.rename(columns={'rank_x': 'rank_home', 'country_full_x': 'country_full_home', 'country_abrv_x': 'country_abrv_home','total_points_x': 'total_points_home', 'previous_points_x': 'previous_points_home','rank_change_x': 'rank_change_home', 'confederation_x': 'confederation_home','rank_date_x': 'rank_date', 'rank_y': 'rank_away', 'country_full_y': 'country_full_away','country_abrv_y': 'country_abrv_away', 'total_points_y': 'total_points_away','previous_points_y': 'previous_points_away', 'rank_change_y': 'rank_change_away','confederation_y': 'confederation_away', 'rank_date_y': 'rank_date_away'}, inplace=True)

    # drop useless and duplicated columns
    reraBase.drop(columns={'rank_date_away', 'country_full_home', 'country_abrv_home', 'confederation_home', 'country_full_away','country_abrv_away', 'confederation_away', 'previous_points_home', 'previous_points_away'},inplace=True)

    # drop the data which have empty field (here we decided to drop those because it represents only 8.45% of the dataset)
    reraBase.dropna(inplace=True)

    return reraBase


""" 
/ remove teams that aren't part of the FIFA in 2023 and teams that have no rank 
/ param rankingData : the dataset rankingData
/ parmam resultsData : the dataset resultsData
"""
def remove_teams_not_in_FIFA_in_2023(rankingData, resultsData):

    # a set of all country names in resultsData
    home_team_and_away_team = set(resultsData.home_team).union(set(resultsData.away_team))
    # a set of all country names in rankingDate
    country_full = set(rankingData.country_full)

    # drop the data of the countries who are not member of the FIFA in 2023 because matches we will have to predict will take place in a FIFA championships
    rankingData = rankingData[~rankingData['country_full'].isin(country_full - fifa_member_countries)]
    # drop the countries who are not member of the FIFA and drop the countries that have no rank
    resultsData = resultsData[~resultsData['home_team'].isin(home_team_and_away_team - country_full)]
    resultsData = resultsData[~resultsData['away_team'].isin(home_team_and_away_team - country_full)]


# ---- main function ----------------------------------------------------------------

def preprocessing():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(SCRIPT_DIR)

    # dataset loading
    rankingData = pd.read_csv(f"{CSV_PATH}/ranking.csv")
    resultsData = pd.read_csv(f"{CSV_PATH}/results.csv")
    lastRankingData = pd.read_csv(f"{CSV_PATH}/last_rank.csv")  # made by us to get the last fifa world ranks that aren't in "ranking.csv"

    # concat both ranking dataset to get all the ranks in only one dataset
    rankingData = pd.concat([rankingData, lastRankingData], axis=0, ignore_index=True)

    # delete the matches for which there hadn't yet fifa world ranks
    indexNames = resultsData[resultsData["date"] < "1992-12-31"].index
    resultsData.drop(indexNames, inplace=True)

    # delete the columns "country" & "city" because we already know thoses columns are useless and if we keep those, we will have to rename more countries
    resultsData = resultsData.drop(columns=['country','city'])

    # rename the countries
    rankingData['country_full'] = rankingData['country_full'].replace(replacementsRankingName)

    # drop the data of the countries who are not member of the FIFA in 2023 because matches we will have to predict will take place in a FIFA championships
    remove_teams_not_in_FIFA_in_2023(rankingData, resultsData)

    # resampling of the dataset rankingData
    rankingDataResample = resampling(rankingData)

    # merging both dataset in only one dataset(reraBase) to have the current world ranks of each team in each matches
    reraBase = merging(rankingDataResample, resultsData)

    # add the target field (a binary class, 0 if home_team win and 1 if away_team win)
    reraBase['winner'] = reraBase.apply(lambda row: who_win(row), axis=1)

    # convert the string field tournament into an integer field
    reraBase['tournament'] = reraBase.apply(lambda row: get_rank_tournament(row['tournament']), axis=1)

    # convert the boolean field neutral into an integer field
    reraBase['neutral'] = reraBase['neutral'].astype(int)

    # create our final dataset without the three first years of the reraBase dataset
    # it will avoid that the new features added won't be for exemple equal 0 in the case we need the average of the 5 previous match
    rera = reraBase.copy(deep=True)
    rera = rera[rera['date'] >= "1996-01-01"]

    # add the new features
    rera = add_features(rera, reraBase)

    # delete the useless features
    rera.drop(columns={'home_team', 'away_team', 'rank_date'}, inplace=True)

    return rera, reraBase, rankingData


