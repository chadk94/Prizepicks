# standard imports
import math
import csv
import os.path as op

# 3rd party imports
import numpy as np
import pandas as pd
import keras

# local imports
import modeling.MLwork as MLwork
import prizepickslines
import dataset.MLSdata as MLSdata
import dataset.datawork as datawork


MLS = ['Opponent_Atlanta Utd', 'Opponent_Austin FC',
       'Opponent_CF Montréal', 'Opponent_Charlotte FC',
       'Opponent_Chicago Fire', 'Opponent_Colorado Rapids',
       'Opponent_Columbus Crew', 'Opponent_D.C. United',
       'Opponent_FC Cincinnati', 'Opponent_FC Dallas',
       'Opponent_Houston Dynamo', 'Opponent_Inter Miami', 'Opponent_LA Galaxy',
       'Opponent_Los Angeles FC', 'Opponent_Minnesota Utd',
       'Opponent_NY Red Bulls', 'Opponent_NYCFC', 'Opponent_Nashville',
       'Opponent_New England', 'Opponent_Orlando City',
       'Opponent_Philadelphia', 'Opponent_Portland Timbers',
       'Opponent_Real Salt Lake', 'Opponent_San Jose', 'Opponent_Seattle',
       'Opponent_Sporting KC', 'Opponent_Toronto FC', 'Opponent_Vancouver']

# SerieA= ADD LATER
# EPL= ADDLATER
# LaLiga= ADDLATER
# Bundesliga= ADDLATER


def playergen(data, league, Playername, Opponent, Home):
    '''Generates a Player entry for the model to predict off of'''
    filter = data['Name'] == Playername
    playerdata = data.where(filter)
    playerdata = playerdata.dropna()
    rolling_shotavg = np.mean(playerdata['Shots'][-5:].astype(float))
    rolling_Minavg = np.mean(playerdata['Min'][-5:].astype(float))
    rolling_xGavg = np.mean(playerdata['xG'][-5:].astype(float))
    rolling_Touchesavg = np.mean(playerdata['Touches'][-5:].astype(float))
    rolling_passavg = np.mean(playerdata['Cmp'][-5:].astype(float))
    season_shotavg = playerdata['Shots'].astype(float).mean()
    season_Minavg = playerdata['Min'].astype(float).mean()
    season_xGavg = playerdata['xG'].astype(float).mean()
    season_Touchesavg = playerdata['Touches'].astype(float).mean()
    season_passavg = playerdata['Cmp'].astype(float).mean()
    if Home == True:
        away = 0
        home = 1
    else:
        away = 1
        home = 0
    opponents = league
    player = [rolling_shotavg, rolling_Minavg, rolling_xGavg,
              rolling_Touchesavg, rolling_passavg, season_shotavg,
              season_Minavg, season_xGavg, season_Touchesavg, season_passavg,
              away, home]
    for i in range(len(opponents)):
        if Opponent in opponents[i]:
            player.append(1)
        else:
            player.append(0)
    player.append(0)  # no start
    player.append(1)  # start
    player.append(0)  # weird star start

    return player


# todo create function to translate lines pulled into and feed into model usng player gen
# todo pull enough data on line too make this categorization? seems difficult.
# need to find a good way to specify league of players, I guess save dictionaries of teams for each laegue we run this on


def linetoprojection(line, league, data):
    line = line[1]
    playerHome = playergen(data, league, line[2], line[5], True)
    playerAway = playergen(data, league, line[2], line[5], False)
    model = keras.models.load_model("MLSmodel.sav")
    playerHome = np.asarray(playerHome)
    playerHome = playerHome.reshape(1, -1)
    playerAway = np.asarray(playerAway)
    playerAway = playerAway.reshape(1, -1)
    if math.isnan(playerHome[0][0]):
        print("Nan value found for player", line[2], "likely league wrong")
        return
    Home = model.predict(playerHome[0])
    Away = model.predict(playerAway[0])
    numbertocompare = line[0]
    print("for ", line.attributes.name, " the line is ", numbertocompare,
          "and we project ", Home, " if home and ", Away, " if Away")


if __name__ == '__main__':
    lines = prizepickslines.getLines('SOCCER', "Shots")
    # MLSdata.get_urls("https://fbref.com/en/comps/22/Major-League-Soccer-Stats")  #RUN THIS ONCE
    data = (datawork.dataload(op.join('data', 'MLSdata.csv')))
    clean = datawork.clean(data)
    # model=MLwork.playerprojection(clean) #RUN WHEN YOU NEED TO GENERATE A MODEL
    for line in lines.iterrows():
        linetoprojection(line, MLS, data)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
