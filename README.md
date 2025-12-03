# Advanced stock forecasting with Bidirectional LSTM + technical indicators, deployed on Hugging Face Spaces with Streamlit UI.

📈# 𝗔𝗱𝘃𝗮𝗻𝗰𝗲𝗱 𝗦𝘁𝗼𝗰𝗸 𝗣𝗿𝗶𝗰𝗲 𝗙𝗼𝗿𝗲𝗰𝗮𝘀𝘁𝗶𝗻𝗴
𝗕𝗶𝗱𝗶𝗿𝗲𝗰𝘁𝗶𝗼𝗻𝗮𝗹 𝗟𝗦𝗧𝗠 • 𝗧𝗲𝗰𝗵𝗻𝗶𝗰𝗮𝗹 𝗜𝗻𝗱𝗶𝗰𝗮𝘁𝗼𝗿𝘀 • 𝗠𝘂𝗹𝘁𝗶-𝗦𝘁𝗲𝗽 𝗧𝗶𝗺𝗲 𝗦𝗲𝗿𝗶𝗲𝘀 𝗣𝗿𝗲𝗱𝗶𝗰𝘁𝗶𝗼𝗻 • 𝗗𝗲𝗽𝗹𝗼𝘆𝗲𝗱 𝗼𝗻 𝗛𝘂𝗴𝗴𝗶𝗻𝗴 𝗙𝗮𝗰e

🔍 𝗢𝘃𝗲𝗿𝘃𝗶𝗲𝘄

This project implements a 𝗺𝘂𝗹𝘁𝗶𝘃𝗮𝗿𝗶𝗮𝘁𝗲 𝗱𝗲𝗲𝗽 𝗹𝗲𝗮𝗿𝗻𝗶𝗻𝗴 model for forecasting future stock price movements using:
•  𝗕𝗶𝗱𝗶𝗿𝗲𝗰𝘁𝗶𝗼𝗻𝗮𝗹 𝗟𝗦𝗧𝗠 𝗮𝗿𝗰𝗵𝗶𝘁𝗲𝗰𝘁𝘂𝗿𝗲
•  𝗢𝗛𝗟𝗖𝗩 (𝗢𝗽𝗲𝗻, 𝗛𝗶𝗴𝗵, 𝗟𝗼𝘄, 𝗖𝗹𝗼𝘀𝗲, 𝗩𝗼𝗹𝘂𝗺𝗲) 𝗺𝗮𝗿𝗸𝗲𝘁 𝗱𝗮𝘁𝗮
•  𝗧𝗲𝗰𝗵𝗻𝗶𝗰𝗮𝗹 𝗶𝗻𝗱𝗶𝗰𝗮𝘁𝗼𝗿𝘀 (𝗦𝗠𝗔, 𝗘𝗠𝗔, 𝗥𝗦𝗜, 𝗠𝗔𝗖𝗗, 𝗕𝗼𝗹𝗹𝗶𝗻𝗴𝗲𝗿 𝗕𝗮𝗻𝗱𝘀, 𝗥𝗲𝘁𝘂𝗿𝗻𝘀)
•  𝗠𝘂𝗹𝘁𝗶-𝘀𝘁𝗲𝗽 𝗳𝗼𝗿𝘄𝗮𝗿𝗱 𝗽𝗿𝗲𝗱𝗶𝗰𝘁𝗶𝗼𝗻 (𝗡-𝗱𝗮𝘆 𝗵𝗼𝗿𝗶𝘇𝗼𝗻)

The model is deployed as a fully interactive web app where users can enter any stock ticker (AAPL, TSLA, RELIANCE.NS, BTC-USD, etc.) and generate live forecasts based on the most recent market data pulled from Yahoo Finance.


🚀 𝗟𝗶𝘃𝗲 𝗗𝗲𝗺𝗼

Platform	Link
🌐 Hugging Face App  https://huggingface.co/spaces/abhishekgupta01/Advanced-Stock-Price-Forecasting-using-Bidirectional-LSTM-Technical-Indicators

📦 GitHub Repo  https://github.com/Abhishek-NR01/Advanced-Stock-Price-Forecasting-using-Bidirectional-LSTM-Technical-Indicators


🧠 Model Architecture
Input → Bidirectional LSTM (64 units) → Dropout
      → Bidirectional LSTM (64 units) → Dense (64, ReLU)
      → Dense (Forecast Horizon)


𝗪𝗵𝘆 𝗕𝗶𝗟𝗦𝗧𝗠?

Financial time series patterns are not purely sequential — future signals may correlate with past volatility ranges. Bidirectional LSTMs allow the model to learn dependencies in both forward and backward temporal directions.


⚙️ 𝗙𝗲𝗮𝘁𝘂𝗿𝗲𝘀

✔ Real-time stock data ingestion via 𝘆𝗳𝗶𝗻𝗮𝗻𝗰𝗲
✔ Automated feature engineering via technical indicators
✔ Scaled inference pipeline with persisted transformers
✔ Multi-step forecasting (predicts multiple future business days)
✔ Interactive interface built using 𝗦𝘁𝗿𝗲𝗮𝗺𝗹𝗶𝘁
✔ Candlestick + forecast overlays using Plotly
✔ Exportable prediction table


🧰 𝗧𝗲𝗰𝗵 𝗦𝘁𝗮𝗰𝗸
   Category	                                  Tools
    Model	                    TensorFlow / Keras (Bidirectional LSTM)
    Data	                          Yahoo Finance (yfinance)
    Deployment	                  Streamlit + Hugging Face Spaces
    Feature                   Engineering	Pandas, NumPy, TA Indicators
    Visualization                      	Plotly, Matplotlib
    Serialization        	Pickle (joblib), .h5 weights, metadata dictionary

    
📦 𝗜𝗻𝘀𝘁𝗮𝗹𝗹𝗮𝘁𝗶𝗼𝗻
Clone the repository:

git clone https://github.com/Abhishek-NR01/Advanced-Stock-Price-Forecasting-using-Bidirectional-LSTM-Technical-Indicators
cd Advanced-Stock-Price-Forecasting-using-Bidirectional-LSTM-Technical-Indicators


Install dependencies:
pip install -r requirements.txt


Run the app locally:
    streamlit run app.py


📊 𝗦𝗮𝗺𝗽𝗹𝗲 𝗢𝘂𝘁𝗽𝘂𝘁

🔹 Price trajectory with prediction overlay
🔹 Candlestick visualization + model trend line
🔹 Forecast table including business-day aligned values
🔹 Growth % metrics (next session vs. full horizon)


🧪 𝗧𝗿𝗮𝗶𝗻𝗶𝗻𝗴 𝗡𝗼𝘁𝗲𝗯𝗼𝗼𝗸
Model retraining can be executed from:

Stock_Price_Prediction_using_Bidirectional_LSTM.ipynb


This notebook includes:

Data collection
Feature engineering
Training loop
Model evaluation & saving
Scaler and metadata persistence


📈 𝗣𝗹𝗮𝗻𝗻𝗲𝗱 𝗘𝗻𝗵𝗮𝗻𝗰𝗲𝗺𝗲𝗻𝘁𝘀

📌 Add sentiment features (news headlines, earnings reports)
📌 Implement walk-forward validation / rolling retraining
📌 Support multi-asset correlation forecasting
📌 Export: CSV, PDF report, Telegram/WhatsApp signal bot
📌 Compare vs. Prophet, ARIMA, TCN, and Transformer-based models


⚠️ 𝗗𝗶𝘀𝗰𝗹𝗮𝗶𝗺𝗲𝗿

This project is 𝗳𝗼𝗿 𝗲𝗱𝘂𝗰𝗮𝘁𝗶𝗼𝗻𝗮𝗹 𝗮𝗻𝗱 𝗿𝗲𝘀𝗲𝗮𝗿𝗰𝗵 𝗽𝘂𝗿𝗽𝗼𝘀𝗲𝘀 𝗼𝗻𝗹𝘆.
It is 𝗻𝗼𝘁 𝗳𝗶𝗻𝗮𝗻𝗰𝗶𝗮𝗹 𝗮𝗱𝘃𝗶𝗰𝗲 and should not be used for live trading without further evaluation, risk modeling, and validation.


⭐ 𝗖𝗼𝗻𝘁𝗿𝗶𝗯𝘂𝘁𝗲

Pull requests and feature suggestions are welcome.
If you'd like to collaborate on improving the forecasting engine or expanding it into a production-grade quant framework, feel free to open an issue.


🏷 𝗔𝘂𝘁𝗵𝗼𝗿

𝗕𝘂𝗶𝗹𝘁 𝗯𝘆: Abhishek Kumar Gupta
📬 Machine Learning & Quant Finance Enthusiast
