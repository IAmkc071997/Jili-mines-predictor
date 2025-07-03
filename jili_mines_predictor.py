
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap
import numpy as np
import matplotlib.pyplot as plt

# Helper function: create new features
def make_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df['win_rate_10'] = df['outcome'].rolling(10, min_periods=1).mean()
    df['risk_level'] = df['num_mines'] / df['board_size']
    return df.dropna()

# Page title
st.title('🔍 Jili Mines Predictor Tool')
st.markdown('Upload your gameplay CSV to get prediction insights!')

# Upload the CSV file
file = st.file_uploader('📁 Upload your Jili game CSV', type='csv')

# If a file is uploaded...
if file:
    raw = pd.read_csv(file)
    data = make_features(raw)

    X = data[['num_mines', 'win_rate_10', 'risk_level']]
    y = data['outcome']

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Predict probability for the next round
    pred_prob = model.predict_proba(X.tail(1))[0, 1]

    st.metric('🎯 Prediction Score', f'{pred_prob * 100:.1f} %')

    # Line chart of payouts
    st.subheader('📈 Payout Trend')
    st.line_chart(data.set_index('timestamp')['payout'])

    # Heatmap of clicks
    st.subheader('🧨 Mine-Click Heatmap')
    heat = np.zeros((5,5))
    for seq in data['clicked_sequence']:
        for idx in map(int, seq.split(',')):
            heat[idx//5, idx%5] += 1
    fig, ax = plt.subplots()
    c = ax.imshow(np.log1p(heat), cmap='YlOrRd')
    plt.title('Click Frequency Heatmap (log scale)')
    st.pyplot(fig)

    # Explainable AI
    st.subheader('📊 Top Prediction Factors')
    explainer = shap.Explainer(model)
    shap_values = explainer(X.tail(1))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.plots.bar(shap_values, show=False)
    st.pyplot()
