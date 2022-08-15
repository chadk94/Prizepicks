import pandas as pd

def dataload(file):
    with open(file) as f:
        print(f)
    data=pd.read_csv(file,encoding='UTF-8')
    data=data.dropna()
    return data
def clean(data):
    multiindex=data[['Name']]
    data.index = pd.MultiIndex.from_frame(multiindex)
    data.sort_index(inplace=True)
    data['rolling_shotavg'] = data.groupby(level=0)['Shots'].rolling(window=5, min_periods=1).mean().values
    data['rolling_Minavg'] = data.groupby(level=0)['Min'].rolling(window=5, min_periods=1).mean().values
    data['rolling_xGavg'] = data.groupby(level=0)['xG'].rolling(window=5, min_periods=1).mean().values
    data['rolling_Touchesavg'] = data.groupby(level=0)['Touches'].rolling(window=5, min_periods=1).mean().values
    data['rolling_passavg'] = data.groupby(level=0)['Cmp'].rolling(window=5, min_periods=1).mean().values
    data['season_shotavg'] = data.groupby(level=0)['Shots'].rolling(window=30,min_periods=1).mean().values
    data['season_Minavg'] = data.groupby(level=0)['Min'].rolling(window=30,min_periods=1).mean().values
    data['season_xGavg'] = data.groupby(level=0)['xG'].rolling(window=30,min_periods=1).mean().values
    data['season_Touchesavg'] = data.groupby(level=0)['Touches'].rolling(window=30,min_periods=1).mean().values
    data['season_passavg'] = data.groupby(level=0)['Cmp'].rolling(window=30,min_periods=1).mean().values
    data=data.dropna()
    filter1 = data['Comp'] == 'MLS'
    filter2 = data['Round'] == 'Regular Season'
    data = data.where(filter1 & filter2)
    cleaned=data.drop(columns=['Result','Squad','Report','Comp','Round','Pos'])#todo reimplementPOs
    cleaned=cleaned.drop(columns=['Day','Min','Gls','Ast','PK','PKatt', 'SoT', 'Yellows', 'Reds',
       'Touches', 'Press', 'Tackles', 'Interceptions', 'Blocks', 'xG', 'npxG',
       'xA', 'SCA', 'GCA', 'Cmp', 'Att', 'Cmp%', 'Progpass','Carries', 'Prog',
       'Succ', 'Att.1'])
    cleaned=pd.get_dummies(cleaned, columns=['Venue','Opponent','Start'])
    cleaned=cleaned.dropna()
    cleaned.index.names=['Name']
    test=cleaned
    return test
    #return cleaned
#todo run on dummies for plyer venue day squad opponent start and rolling values.

#setup neural net to predict shots
#create interface to compare vs.prizepicks