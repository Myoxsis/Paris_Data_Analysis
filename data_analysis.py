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
import seaborn as sns

pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 999)

#%%

df = pd.read_csv('real_estate_clean.csv')
df = df.drop(['Unnamed: 0'], axis=1)



#%%

df.head()
# %%



# %%

sns.pairplot(df)

# %%

sns.boxplot(df['sqrt_tag'])

# %%

# %%

#%%

x = df['city_a'].value_counts().values
y = df['city_a'].value_counts().index

plt.figure(figsize=(20,8))
plt.xticks(rotation=90)
plt.bar(y,x)

#%%

class_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
mapping = {val: i for i, val in enumerate(class_order)}

x = df['energy_class'].value_counts().sort_index(level= class_order).values
y = df['energy_class'].value_counts().sort_index(level= class_order).index

plt.figure(figsize=(20,8))
plt.xticks(rotation=90)
plt.bar(y,x)

# %%

class_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
mapping = {val: i for i, val in enumerate(class_order)}

x = df['ges_class'].value_counts().sort_index(level= class_order).values
y = df['ges_class'].value_counts().sort_index(level= class_order).index

plt.figure(figsize=(20,8))
plt.xticks(rotation=90)
plt.bar(y,x)

# %%

# %%

#%%

img=mpimg.imread("map.PNG")

cmap          = matplotlib.cm.get_cmap('YlOrRd')
my_cmap       = cmap(np.arange(cmap.N))
my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
my_cmap       = matplotlib.colors.ListedColormap(my_cmap)

x_min = 2.2041 #min(df['long']) - 0.100
x_max = 2.3207 #max(df['long']) + 0.100
y_min = 48.8732 #min(df['lat']) - 0.100
y_max = 48.9310 #max(df['lat']) + 0.100
colors = {1.:'red', 2.:'green'}

extent = (x_min, x_max, y_min, y_max)

plt.Figure(figsize=(679,515), dpi=50)
fig = plt.gcf()
fig.set_size_inches(9.43,7.15)
fig.set_dpi(500)
ax = plt.gca()
ax.set_aspect('auto')
plt.title('Courbevoie')

plt.imshow(img, aspect='auto', extent = extent, cmap='gray') # Comment to remove background


plt.scatter(df['long'], df['lat'], c=df['transaction_type'].map(colors), s=5)

plt.xlim((x_min,x_max))
plt.ylim((y_min,y_max))
plt.xticks(size=5)
plt.yticks(size=5)

colormap = np.array(['r', 'g', 'b'])

plt.show()


#%%

colors = {1.:'red', 2.:'green'}

fig = go.Figure()
fig.add_trace(go.Scatter(x= df['long'], y=df['lat'], mode='markers',
                marker=dict(color=df['transaction_type'].map(colors)),
                hovertemplate =
                    '<b>%{text}</b>',
                text = df['titre']))
fig.update_layout(title="Real Estate - Courbevoie Extended Scope",
                 xaxis_title="Lat",
                 yaxis_title="Long")
fig.update_xaxes(range=[2.2041, 2.3207])
fig.update_yaxes(range=[48.8732, 48.9310])
fig.show()


#%%

df['date'].value_counts()

#%%



# %%

date_dict = {}
i = 0
for day in df['date'].sort_values(ascending=True).unique():
    date_dict[i] = day
    i += 1

def animate_plot(x):
    x_min = 2.2041
    x_max = 2.3207
    y_min = 48.8732
    y_max = 48.9310
    colors = {1.:'red', 2.:'green'}

    date = date_dict[x]
    print(date)
    dt = df.loc[df['date'] == date, :]
    ax.scatter(dt['long'], dt['lat'], c=dt['transaction_type'].map(colors), marker='o', s=5)

    extent = (x_min, x_max, y_min, y_max)

    plt.imshow(img, aspect='auto', extent = extent, cmap='gray')

    ax.set_xlim((x_min,x_max))
    ax.set_ylim((y_min,y_max))

    ax.set_title('Hauts-de-Seine \nPublication Date = ' + str(date)[:10])


# Test animation
fig = plt.figure()
ax = plt.axes()
anim = animation.FuncAnimation(fig, animate_plot, interval = 1000, frames = len(df['date'].sort_values(ascending=True).unique()),
repeat=False)

# Saving the Animation
f = r"animate_func1.gif"
writergif = animation.PillowWriter(fps=len(df['date'].sort_values(ascending=True).unique())/6)
anim.save(f, writer=writergif)

# %%
df.info()

# %%



#%%

date_dict

#%%

df['date'].sort_values(ascending=True).unique()

#%%



 # Comment to remove background


plt.scatter(df['long'], df['lat'], c=df['transaction_type'].map(colors), s=5)

#%%



#%%

df.head()

#%%

#%%

# geotree integration to be done 
t = df.loc[df['quartier_a'].isna(), :]
t['city_a'].value_counts(dropna=False)

#%%

t = df.loc[df['city_a'].isna(), :]
t['city'].value_counts(dropna=False)
#%%

#%%

matrix = df.corr()

plt.figure(figsize=(16,12))

# Create a custom diverging palette

cmap = sns.diverging_palette(250, 15, s=75, l=40,
                             n=9, center="light", as_cmap=True)

_ = sns.heatmap(matrix, center=0, annot=True, 
                fmt='.2f', square=True, cmap=cmap)

#%%

df.head()

#%%


#%%

# Custom-defined Label Encoder

def clean_transaction(df):
    df['transaction_type'] = df['transaction_type'].replace(['Location', 'Vente'], [1., 2.])
    return df

def clean_property(df):
    df['property_type'] = df['property_type'].replace(['Appartement', 'Studio', 'Garage, Parking', 'Maison', 'Bureaux et Locaux Professionnels', 'Local Commercial', 'Fonds de Commerce', 'Immeuble', 'Chambre', 'Péniche'], 
                                                [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    return df

def clean_ges(df):
    df['ges_class'] = df['ges_class'].replace(['A', 'B', 'C', 'D', 'E', 'F', 'G'], 
                                                [7., 6., 5., 4., 3., 2., 1.])
    df['ges_class'] = df['ges_class'].fillna(0)
    return df

def clean_energy(df):
    df['energy_class'] = df['energy_class'].replace(['A', 'B', 'C', 'D', 'E', 'F', 'G'], 
                                                [7., 6., 5., 4., 3., 2., 1.])
    df['energy_class'] = df['energy_class'].fillna(0)
    return df

def clean_tag(df):
    df['bedroom_tag'] = df['bedroom_tag'].fillna(0)
    df['sqrt_tag'] = df['sqrt_tag'].fillna(0)
    df['piece_tag'] = df['piece_tag'].fillna(0)
    return df

df = clean_transaction(df)
df = clean_energy(df)
df = clean_ges(df)
df = clean_property(df)
df = clean_tag(df)


#%%

df.to_csv('real_estate_clean_ML.csv')