
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Prediction (ML-Prophet)')
st.write('** Made by Aryan Tanwar for education purpose (NPS HSR)**')
st.write('Only 130 popular stocks ticker included')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'WISH', 'WISH', 'CLOV', ' WEN', 'BTC-USD', 'GOEV', 'GME', 'FSLY',
'AHT' , 'SENS', 'RIDE', 'KODK', 'ASTS', 'WKHS', 'MU', 'VLDR', 'APPN', 'AC.TO', 'LOTZ', 'AMZN', 'OPEN', 'NOKPF',
'SOFI', 'HYLN', 'RKT', 'SDC', 'SPWR', 'SIRI', 'ROOT', 'OTRK', 'WOOF', 'ABBV','ABT','ACN','ADBE','AIG',
'AMGN','AVGO','AXP','BA','BAC','BIIB','BK','BKNG','BLK','BMY','BRKB','C','CAT','CHTR','CL','CMCSA','COF',
'COP','COST','CRM','CSCO','CVS','CVX','DD','DHR','DIS','DOW','DUK','EMR','EXC','F','FB','FDX','GD','GE',
'GILD','GM','GOOGL','GS','HD','HON','IBM','INTC','JNJ','JPM','KHC','KO','LIN','LLY','LMT','LOW','MA','MCD',
'MDLZ','MDT','MET','MMM','MO','MRK','MS','MSFT','NEE','NFLX','NKE','NVDA','ORCL','PEP','PFE','PG','PM','PYPL',
'QCOM','RTX','SBUX','SO','SPG','T','TGT','TMO','TMUS','TSLA','TXN','UNH','UNP','UPS','USB','V','VZ','WBA','WFC'
,'WMT','XOM')
selected_stock = st.sidebar.selectbox('Select dataset for prediction', stocks)

n_years = st.sidebar.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

plot_raw_data()

df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Prediction data')
st.write(forecast.tail())

st.write(f'Prediction plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Prediction components")
fig2 = m.plot_components(forecast)
st.write(fig2)