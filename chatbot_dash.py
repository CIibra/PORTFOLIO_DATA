# src/chatbot_dash.py
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import requests
import pandas as pd

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Chatbot Citoyen Intelligent"),
    dcc.Input(id="user_input", type="text", placeholder="Posez votre question..."),
    html.Button("Envoyer", id="send_btn"),
    html.Div(id="chat_output"),
    dcc.Interval(id="interval", interval=5000, n_intervals=0),  # refresh toutes les 5s
    dcc.Graph(id="stats_graph")
])

@app.callback(
    Output("chat_output", "children"),
    Input("send_btn", "n_clicks"),
    State("user_input", "value")
)
def update_chat(n, question):
    if not n or not question:
        return ""
    r = requests.post("http://127.0.0.1:8000/ask", json={"question": question})
    answer = r.json()["answer"]
    return f"Vous: {question} | Bot: {answer}"

@app.callback(
    Output("stats_graph", "figure"),
    Input("interval", "n_intervals")
)
def update_graph(n):
    r = requests.get("http://127.0.0.1:8000/stats")
    stats = r.json()
    if not stats:
        return px.bar(title="Aucune question posée")
    df = pd.DataFrame({"minute": list(stats.keys()), "questions": list(stats.values())})
    fig = px.bar(df, x="minute", y="questions", title="Questions posées par minute")
    return fig

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)