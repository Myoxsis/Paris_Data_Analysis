#%%

# Improve City => Geotree library to update


#%%

import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from datetime import datetime
import matplotlib # Graphical Visualisation
import matplotlib.dates as pltdt
import matplotlib.image as mpimg
import matplotlib.patches
import plotly.graph_objects as go 
# Importing Packages
import numpy as np
from matplotlib import animation

from geotree2 import GeoTree

pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 999)

#%%


def get_property_type(x):
    r = ''
    if re.findall('studio', x) != []:
        r = 'Studio'
    elif re.findall('appartement', x) != []:
        r = 'Appartement'
    elif re.findall('maison', x) != []:
        r = 'Maison'
    elif re.findall('hotel', x) != []:
        r = 'Hotel'
    elif re.findall('garage, parking', x) != []:
        r = 'Garage, Parking'
    elif re.findall('bureaux et locaux professionnels', x) != []:
        r = 'Bureaux et Locaux Professionnels'
    elif re.findall('local commercial', x) != []:
        r = 'Local Commercial'
    elif re.findall("local d'act", x) != []:
        r = 'Local Commercial'
    elif re.findall('fonds de commerce', x) != []:
        r = 'Fonds de Commerce'
    elif re.findall('péniche', x) != []:
        r = 'Péniche'
    elif re.findall('Vente immeuble', x) != []:
        r = 'Immeuble'
    elif re.findall('Vente chambre', x) != [] or re.findall('Location chambre', x) != []:
        r = 'Chambre'
    else:
        r = np.nan
    return r

def get_publication_date(x):
    r = str(x).split(' / ')[1]
    return r

def get_reference_id(x):
    r = str(x).split(' / ')[0]
    return r

def clean_price_data(x):
    try:
        r = x.replace('.', '').replace('€', '').replace(' ','')
        r = float(r)
    except:
        r = np.nan
    return r

def clean_ref(x):
    r = x.lstrip('Réf. : ')
    return r

def convert_text_to_date(x):
    x = x.strip(' ')
    y = x[-4:]
    d = re.findall("janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre", x)
    if not d:
        m = 0
    elif d[0] == 'janvier':
        m = 1
    elif d[0] == 'février':
        m = 2
    elif d[0] == 'mars':
        m = 3
    elif d[0] == 'avril':
        m = 4
    elif d[0] == 'mai':
        m = 5
    elif d[0] == 'juin':
        m = 6
    elif d[0] == 'juillet':
        m = 7
    elif d[0] == 'août':
        m = 8
    elif d[0] == 'septembre':
        m = 9
    elif d[0] == 'octobre':
        m = 10
    elif d[0] == 'novembre':
        m = 11
    elif d[0] == 'décembre':
        m = 12
    else:
        m = 0 

    m = str(f'0{m}') if m < 10 else str(m)

    date_number = x[:-(len(x)-len(str(d[0])) + 1 + len(str(y)) + 1)+1]
    f_date = datetime.strptime(f'{date_number}/{m}/{y}', '%d/%m/%Y')
    return f_date

def clean_piece_tag(x):
    r = str(x).split(' ')[0]
    if r == 'nan':
        r = np.nan
    else:
        r = float(r)
    return r

def clean_bedroom_tag(x):
    r = str(x).split(' ')[0]
    if r == 'nan':
        r = np.nan
    else:
        r = float(r)
    return r

def clean_sqrt(x):
    x = x.replace("'", '').strip('][').split(', ')
    for elem in x:
        s_d = re.findall('^([0-9]+) m²', elem)
        if s_d != []:
            r = s_d[0]
    try:
        r = r.replace(' m²', '').replace(',', '.')
    except:
        r = np.nan
    return r

def clean_terrain_sqrt(x):
    x = x.replace("'", '').strip('][').split(', ')
    for elem in x:
        s_d = re.findall('Terrain ([0-9]+) m²', elem)
        if s_d != []:
            r = s_d[0]
    try:
        r = r.replace(' m²', '').replace(',', '.')
    except:
        r = np.nan
    return r

def clean_psqrt(x):
    x = x.replace("'", '').strip('][').split(', ')
    try:
        r = x[3]
        r = r.replace(' € le m²', '').replace('.', '').replace(',', '.')
    except:
        r = np.nan
    return r

def get_coordinates(x, type_coord):
    r = x.lstrip('{"center":["').rstrip('"],"zoom":15}').replace('","', ',').replace(']', '')

    lat = r.split(',')[0]
    long = r.split(',')[1]
    
    if type_coord == 'lat':
        return lat
    elif type_coord == 'long':
        return long

def get_transaction_type(x):
    r = x.split(' ')[0]
    if r == 'Vente':
        y = 'Vente'
    elif r == 'Location':
        y = 'Location'
    else:
        y = np.nan
    return y

def get_city(x):
    r = ''
    city_list = ["Vélizy-Villacoublay", "Orvault", "Vertou", "Bordeaux", "Bois-D'arcy", 'Le Chesnay-Rocquencourt', 'Bailly', 'Rueil-Malmaison', 'Saint-Herblain', 'Couëron', 'Fontenay-Le-Fleury', 'Nantes', 'Bougival', 'Sèvres', 'Asnières-Sur-Seine', 'La Garenne-Colombes', 'Colombes', 'Montigny-Le-Bretonneux', 'Issy-Les-Moulineaux', 'Louveciennes', 'Versailles', 'Meudon', 'Marly-Le-Roi' ,'Paris', 'Courbevoie', 'Clichy', 'Levallois-Perret', 'Garches', 'Suresnes', 'Saint-Cloud', 'Puteaux', 'Neuilly-Sur-Seine', 'Boulogne-Billancourt', 'Lyon', 'Toulouse', 'Chaville']
    search_city = '*|'.join(city_list)

    d = re.findall(search_city, x)
    
    if d != []:
        r = d[0]
    else :
       r = np.nan
    return r

# %%

df = pd.read_csv('real_estate_df_geoloc.csv')
df.drop(columns='Unnamed: 0', axis=1, inplace=True)
df.head()

# Feature engineering
df['property_type'] = df['titre'].apply(lambda x : get_property_type(x))
df['city'] = df['titre'].apply(lambda x : get_city(x))
df['ref'] = df['date'].apply(lambda x : get_reference_id(x))
df['date'] = df['date'].apply(lambda x : get_publication_date(x))
df['lat'] = df['geoloc'].apply(lambda x : get_coordinates(x, 'lat'))
df['long'] = df['geoloc'].apply(lambda x : get_coordinates(x, 'long'))
df['transaction_type'] = df['titre'].apply(lambda x : get_transaction_type(x))
df['isBalcony'] = [1. if re.findall('[Bb]alcon', x) != [] else 0. for x in df['desc']]
df['isParking'] = [1. if re.findall('[Pp]arking', x) != [] else 0. for x in df['desc']]
df['isPool'] = [1. if re.findall('[Pp]iscine', x) != [] else 0. for x in df['desc']]
df['isGarden'] = [1. if re.findall('[Jj]ardin|[Cc]our', x) != [] else 0. for x in df['desc']]

# Data Cleaning
df['ref'] = df['ref'].apply(lambda x : clean_ref(x))
df['price'] = df['price'].apply(lambda x : clean_price_data(x))
df['date'] = df['date'].apply(lambda x : convert_text_to_date(x))
df['piece_tag'] = df['piece_tag'].apply(lambda x : clean_piece_tag(x))
df['bedroom_tag'] = df['bedroom_tag'].apply(lambda x : clean_bedroom_tag(x))
df['sqrt_tag'] = df['tag_list'].apply(lambda x : clean_sqrt(x))
df['terrain_sqrt_tag'] = df['tag_list'].apply(lambda x : clean_terrain_sqrt(x))
df['psqrt_tag'] = df['tag_list'].apply(lambda x : clean_psqrt(x))
df['lat'] = pd.to_numeric(df['lat'])
df['long'] = pd.to_numeric(df['long'])

# Drop Duplicated references
df = df.drop_duplicates(subset='ref', keep='last').reset_index(drop=True)


#%%

# Restrict data to La Defense Area
df = df[(df['long'] > 2.2041) & (df['long'] < 2.3207)]
df = df[(df['lat'] > 48.8732) & (df['lat'] < 48.9310)]


# %%
# Geocoding !! can be long to execute
geotree = GeoTree()
df['city_a'] = df.apply(lambda x : geotree.search_geotree(x.long, x.lat)[0], axis=1)
df['quartier_a'] = df.apply(lambda x : geotree.search_geotree(x.long, x.lat)[1], axis=1)


#%%

# geotree integration to be done 
t = df.loc[df['quartier_a'] == '', :]
t['city_a'].value_counts(dropna=False)

#%%

t = df.loc[df['city_a'] == '', :]
t['city'].value_counts(dropna=False)

#%%

df.head()

#%%

y = t.loc[t['city_a'] == '', :]
y['titre'].value_counts()

# %%

y.head()

# %%

df.to_csv('real_estate_clean.csv')

# %%

len(df)



#%%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
# %%
