import prizepickslines
import MLSdata
import csv
import pandas as pd
import datawork
import MLwork
if __name__ == '__main__':
    #prizepickslines.getLines('SOCCER', "Shots")
   #MLSdata.get_urls("https://fbref.com/en/comps/22/Major-League-Soccer-Stats")  #RUN THIS ONCE
    data=(datawork.dataload("MLSdata.csv"))
    player=(["Sat","Home","Los Angeles FC","Sporting KC","Y",0,"Carlos Vela",12/5,405/5,.8/5,236/5,156/5,41/18,1425/18,5.5/18,892/18,527/18])
    clean=datawork.clean(data,player)
    model=MLwork.propbet(clean)
    print (clean.iloc[-1:])
    print("Player ", player[6], " is projected to take ", float(clean.iloc[-1:]['Model Predictions']), " shots")
   # MLwork.playerprojection(clean,"Ola Kamara", "CF Montreal", True, MLwork.simplemodel())
    #with open('MLSdata.csv','w',newline="",encoding="UTF-8") as f:
       # write=csv.writer(f)
       # write.writerows(MLSdata.get_player_data("https://fbref.com/en/players/4501fd21/matchlogs/2022/summary/Zan-Kolmanic-Match-Logs"))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
