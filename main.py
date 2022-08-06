import prizepickslines
import MLSdata
import csv
import numpy as np
import pandas as pd
import datawork
import MLwork
import keras


def playergen(data, Playername, Opponent, Home):
    filter=data['Name']==Playername
    playerdata = data.where(filter)
    playerdata=playerdata.dropna()
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
    opponents = ['Opponent_Atlanta Utd', 'Opponent_Austin FC',
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
    player = [rolling_shotavg, rolling_Minavg, rolling_xGavg,
              rolling_Touchesavg, rolling_passavg, season_shotavg,
              season_Minavg, season_xGavg, season_Touchesavg, season_passavg,
              away, home]
    for i in range(len(opponents)):
        if Opponent in opponents[i]:
            opponents[i] = 1
        else:
            opponents[i] = 0
        player.append(opponents[i])
    player.append(0)
    player.append(1)
    player.append(0)

    return player


if __name__ == '__main__':
    # prizepickslines.getLines('SOCCER', "Shots")
    #MLSdata.get_urls("https://fbref.com/en/comps/22/Major-League-Soccer-Stats")  #RUN THIS ONCE
    data = (datawork.dataload("MLSdata.csv"))
    clean = datawork.clean(data)
    model=MLwork.playerprojection(clean) #RUN WHEN YOU NEED TO GENERATE A MODEL
    model = keras.models.load_model("MLSmodel.sav")
    name="Marcelino Moreno"
    player = playergen(data, "Marcelino Moreno", "Chicago", True)

    # model=MLwork.propbet(clean)
    # print (clean.iloc[-1:])
    player=np.asarray(player)
    player=player.reshape(1,-1)
    print (player)
    print("Player ", name, " is projected to take ", model.predict(player)[0], " shots")
# MLwork.playerprojection(clean,"Ola Kamara", "CF Montreal", True, MLwork.simplemodel())
# with open('MLSdata.csv','w',newline="",encoding="UTF-8") as f:
# write=csv.writer(f)
# write.writerows(MLSdata.get_player_data("https://fbref.com/en/players/4501fd21/matchlogs/2022/summary/Zan-Kolmanic-Match-Logs"))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
