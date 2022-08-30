import streamlit as st
import pandas as pd
import seaborn as sns
import requests
import matplotlib.pyplot as plt
import shap


# path = 'http://127.0.0.1:5000/'
path = 'https://her-my-api.herokuapp.com/'

# On fait un requête GET à notre API Flask pour obtenir le dataset
df = pd.DataFrame(requests.get(path+'raw_dataset/').json())

st.title('DASHBOARD CLIENTS')

threshold = 0.5
sns.set(rc={"axes.facecolor":"black", "axes.grid":False,'xtick.labelsize':14,'ytick.labelsize':14, 'figure.facecolor':'lightgrey', 'legend.facecolor' :'white'})

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(12, 8))

def get_idx():
	return pd.Series(requests.get(path+'get_idx/').json()).sort_values()

id = st.sidebar.selectbox('IDENTIFIANT CLIENT', get_idx())

def get_prediction(id):
	return float(requests.get(path+'predict/?id={}'.format(id)).json())
pred = get_prediction(id)



dur = df.DURATION.loc[df['SK_ID_CURR'] == id]
exts3 = df.EXT_SOURCE_3.loc[df['SK_ID_CURR'] == id]
amtcr  =  df.EXT_SOURCE_2.loc[df['SK_ID_CURR'] == id]
amtgprice = df.AMT_GOODS_PRICE.loc[df['SK_ID_CURR'] == id]


# Write prediction
# st.write("Probabilité de défaut : {}".format(pred))
st.metric('Probabilité de défaut',pred,0.5-pred)

bool_cust = (pred >= threshold)

if bool_cust is False:
   decision = "accepté" 
            
else:
    decision = "rejeté"
st.write("Prêt :", decision)


sns.histplot(ax=ax1, x="DURATION", hue = 'TARGET', data=df, palette='Set2', element="poly", stat="density", common_norm=False)
ax1.axvline(dur[0], 0, 1, color = 'r' if threshold<float(pred) else 'g',linestyle = 'dashed',linewidth=2)
sns.histplot(ax=ax2, x="EXT_SOURCE_3", hue = 'TARGET', data=df, palette='Set2', element="poly", stat="density", common_norm=False, legend = False)
ax2.axvline(exts3[0], 0, 1, color = 'r' if threshold<float(pred) else 'g',linestyle = 'dashed',linewidth=2)
sns.histplot(ax=ax3, x="EXT_SOURCE_2", hue = 'TARGET', data=df, palette='Set2', element="poly", stat="density", common_norm=False, legend = False)
# ax3.set_xlim(0,3e6)
ax3.axvline(amtcr[0], 0, 1, color = 'r' if threshold<float(pred) else 'g',linestyle = 'dashed',linewidth=2)
sns.histplot(ax=ax4, x="AMT_GOODS_PRICE", hue = 'TARGET', data=df, palette='Set2', element="poly", stat="density", common_norm=False, legend = False)
ax4.axvline(amtgprice[0], 0, 1, color = 'r' if threshold<float(pred) else 'g',linestyle = 'dashed',linewidth=2)
ax4.set_xlim(0,2.5e6)
st.pyplot(f, ((ax1, ax2), (ax3, ax4)))

