#with thanksto https://colab.research.google.com/drive/1PoHtZWcy8WaU1hnWmL7eCVUbxzci3-fr#scrollTo=gPK7Y54NmUVA
import pandas as pd
import csv
import time
#updatefiles as necessarry
#lter include scrapingin this
import requests
from bs4 import BeautifulSoup

def get_urls(leagueurl):#break this up ito multiple functions
    output=[]
    headers=['Day','Comp','Round','Venue','Result','Squad','Opponent','Start','Pos','Min','Gls','Ast','PK','PKatt','Shots','SoT','Yellows','Reds','Touches','Press','Tackles','Interceptions','Blocks','xG','npxG','xA','SCA','GCA','Cmp','Att','Cmp%','ProgPass','Carries','Prog','Succ','Att','Report','Name']
    with open('MLSdata.csv','a',newline="",encoding="UTF-8") as f:
        write=csv.writer(f)
        write.writerow(headers)
        while (True):
                page = requests.get(leagueurl)
                if (page.status_code!=429):
                    print ("navigating to league page")
                    break
                else:
                    time.sleep(15)
                print("trying again")
        soup=BeautifulSoup(page.content,'html.parser')
        time.sleep(.5)
        metric_names = []
        for row in soup.findAll('table')[4].tbody.findAll('tr'):
            first_column = row.findAll('th')[0].contents
            metric_names.append(first_column)
        teamcount=0
        for i in metric_names:
            teamcount+=1
            print ("starting team", teamcount, "of ", len(metric_names))
            link=(str(i).split('"')[1])
            while (True):
                page = requests.get(("http://fbref.com" + link))
                if (page.status_code != 429):
                    print("navigating to team page")
                    break
                else:
                    time.sleep(15)
                print("trying again")
            soup = BeautifulSoup(page.content, 'html.parser')
            for row in soup.findAll('table')[0].tbody.findAll('tr'):
                first_column = row.findAll('th')[0].contents
                id=str(first_column).split('/')[3]
                name=str(first_column).split('/')[4].split('"')[0]
                link=("https://fbref.com/en/players/"+id+"/matchlogs/2022/summary/"+name+"-Match-Logs")
                print(link)
                newplayer=get_player_data(link)
                write.writerows(newplayer)
                output.append(newplayer)
    return output
def get_player_data(x):
    url = x
    while (True):
            page = requests.get(url)
            if (page.status_code!=429):
                print ("status code=",page.status_code," printing player to file")
                break
            else:
                time.sleep(15)
            print("trying again")
    soup = BeautifulSoup(page.content, 'html.parser')
    name = [element.text for element in soup.find_all("span")]
    name = name[7]
    table_body = soup.find_all('table')[0].tbody
    rows = table_body.find_all('tr')
    dict=[]
    for row in rows:
        cols = row.find_all('td')
        cols = [x.text.strip() for x in cols]
        cols.append(name)
        dict.append(cols)
    return dict
