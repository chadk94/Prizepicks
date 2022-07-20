import prizepickslines
import MLSdata
import csv
import datawork
if __name__ == '__main__':
    #prizepickslines.getLines('SOCCER', "Shots")
    MLSdata.get_urls("https://fbref.com/en/comps/22/Major-League-Soccer-Stats")  #RUN THIS ONCE
    #print(datawork.dataload("MLSdata.csv"))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
