
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader.data as web
import pandas as pd
import datetime as dtfi

def Graph(x, titles, values):
    df = pd.DataFrame()
    df.index = x
    for z in range(len(titles)):
        df[titles[z]] = values[z]
    print(df)
    plot = df.plot(title= "Graph")
    plot.set_yscale("log")
    plt.xticks(rotation=25)
    plt.savefig("plot.png")

def Download(ticker, source, Start, End):
    if(source[0]=="pdr"):
        print("update")
        web.DataReader(ticker, source[1], Start, End).to_csv(f"Finance_Data/{ticker}.csv")
    if(source[0]=="yf"):
        yf.Ticker(ticker).history(start=Start, end = End).to_csv(f"Finance_Data/{ticker}.csv")

def Load(ticker, new=False):
    if(new):
        print(
            "################################################\n"+
            "################################################\n"+
            "#################  NOT FOUND  ##################\n"+
            "################################################\n"+
            "################################################\n"
        )
    return pd.read_csv(f"Finance_Data/{ticker}.csv")

# ------------------------------------------------------------------------
# --------------------------------DOWNLOAD--------------------------------
# ----------- should mostly be commented unless actively using -----------
# ------------------------------------------------------------------------


# Download("CPIAUCSL", ["pdr","fred"], dt.datetime(2010, 1, 1), dt.datetime(2020, 1, 1))
# Download("NVDA", ["yf"], dt.datetime(2018, 1, 1), dt.datetime(2026, 1, 1))
# Download("US67066GAE4", ["yf"], dt.datetime(2018, 1, 1), dt.datetime(2026, 1, 1))

# ------------------------------------------------------------------------
# ----------------------------------LOAD----------------------------------
# ------------------------------------------------------------------------

# cpi = Load("CPIAUCSL")
rawnvda = Load("NVDA")

# ------------------------------------------------------------------------
# ------------------------------CONFORMATION------------------------------
# ------------------------------------------------------------------------


x = rawnvda["Date"].str.split(' ', n=1).str[0]
nvda = np.array(((rawnvda["High"]+rawnvda["Low"])/2).round(3))
sqnvda = np.array(nvda**2)
sqrtnvda = np.array(nvda**.5)

# ------------------------------------------------------------------------
# ---------------------------------GRAPH----------------------------------
# ------------------------------------------------------------------------


Graph(x, ["nvda", "sqnvda", "sqrtnvda"], [nvda, sqnvda, sqrtnvda])
# Graph(cpi["DATE"], cpi["CPIAUCSL"])

print(2010%14)