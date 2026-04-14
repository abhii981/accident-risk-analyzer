import gradio as gr
import pandas as pd
import numpy as np
import joblib
from fastapi import Request
from fastapi.responses import JSONResponse
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Try importing Gemini, fallback if not available
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ Gemini AI not available. Chat feature will be disabled.")

# ============================================
# 1. LOAD MODELS AND DATA
# ============================================

print("🚀 Loading models and data...")

def load_models():
    """Load all trained models"""
    print("  Loading models...")
    rf = joblib.load('models/rf_model.pkl')
    knn = joblib.load('models/knn_model.pkl')
    lr = joblib.load('models/lr_model.pkl')
    xgb = joblib.load('models/xgb_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return rf, knn, lr, xgb, scaler

def load_data():
    """Load and preprocess data"""
    print("  Loading data...")
    df = pd.read_csv('data/indian_roads_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = pd.to_datetime(df['time']).dt.hour
    df['festival'] = df['festival'].fillna('none')
    return df

# Load everything
rf, knn, lr, xgb, scaler = load_models()
df = load_data()

# Pre-compute statistics
print("  Computing statistics...")
stats = {
    'total_accidents': len(df),
    'avg_risk': df['risk_score'].mean(),
    'fatal_count': (df['accident_severity'] == 'fatal').sum(),
    'fatal_rate': (df['accident_severity'] == 'fatal').mean() * 100,
    'hourly_risk': df.groupby('hour')['risk_score'].mean().to_dict(),
    'city_risk': df.groupby('city')['risk_score'].mean().sort_values(ascending=False).to_dict(),
    'weather_risk': df.groupby('weather')['risk_score'].mean().to_dict(),
}

# ============================================
# 2. GEMINI AI SETUP
# ============================================

if GEMINI_AVAILABLE:
    genai.configure(api_key="AIzaSyC94QgySrGQ0c3XmzQ2WfkJqTXw9th4tnI")
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")

SYSTEM_CONTEXT = """
You are an expert road safety analyst AI assistant for an Indian Road Accident Risk Analyzer.
The dataset contains 20,000 accident records from 8 Indian cities.
Key findings:
- Random Forest achieved 69% accuracy for severity prediction
- Fatal accidents are most common during fog and low visibility
- Peak hours (8-10 AM, 5-8 PM) and weekends show higher risk scores
- Festival periods (Diwali, Holi, Eid) show elevated accident rates
Answer questions about road safety, accident patterns, and this project.
Keep answers concise and helpful.
"""

def chat_with_gemini(message):
    if not GEMINI_AVAILABLE:
        return "Chat feature is currently unavailable."
    try:
        prompt = f"{SYSTEM_CONTEXT}\n\nUser: {message}\nAssistant:"
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"I apologize, but I couldn't process that request. Error: {str(e)}"

# ============================================
# 3. ENCODING MAPS
# ============================================

city_map = {'Mumbai': 0, 'Pune': 1, 'Delhi': 2, 'Chennai': 3,
            'Bangalore': 4, 'Hyderabad': 5, 'Kolkata': 6, 'Chandigarh': 7}
weather_map = {'Clear': 0, 'Rain': 1, 'Fog': 2}
road_map = {'Highway': 0, 'Urban': 1, 'Rural': 2}
vis_map = {'Low': 0, 'Medium': 1, 'High': 2}
traffic_map = {'Low': 0, 'Medium': 1, 'High': 2}
day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
           'Friday': 4, 'Saturday': 5, 'Sunday': 6}
festival_map = {'None': 0, 'Diwali': 1, 'Holi': 2, 'Eid': 3, 'New Year': 4}
cause_map = {'Weather': 0, 'Overspeeding': 1, 'Driver Error': 2, 'Road Condition': 3}

# ============================================
# 4. PREDICTION FUNCTION
# ============================================

def predict_risk(city, weather, road_type, visibility, traffic_density,
                 hour, is_weekend, is_peak_hour, temperature, lanes,
                 traffic_signal, vehicles_involved, casualties,
                 risk_score, day_of_week, festival, cause, model_choice):
    """Fast prediction with risk levels"""
    try:
        # Prepare features
        features = np.array([[
            hour, int(is_weekend), int(is_peak_hour), temperature,
            lanes, int(traffic_signal), vehicles_involved, casualties,
            risk_score, 6,
            weather_map.get(weather, 0),
            road_map.get(road_type, 0),
            vis_map.get(visibility, 0),
            traffic_map.get(traffic_density, 0),
            festival_map.get(festival, 0),
            cause_map.get(cause, 0),
            day_map.get(day_of_week, 0),
            city_map.get(city, 0), 0
        ]])

        # Quick model selection
        if model_choice == 'Random Forest':
            pred = rf.predict(features)[0]
            proba = rf.predict_proba(features)[0]
        elif model_choice == 'XGBoost':
            pred = xgb.predict(features)[0]
            proba = xgb.predict_proba(features)[0]
        elif model_choice == 'KNN':
            features_scaled = scaler.transform(features)
            pred = knn.predict(features_scaled)[0]
            proba = knn.predict_proba(features_scaled)[0]
        else:
            features_scaled = scaler.transform(features)
            pred = lr.predict(features_scaled)[0]
            proba = lr.predict_proba(features_scaled)[0]

        # Calculate risk
        risk_score_val = (proba[0] * 20 + proba[1] * 50 + proba[2] * 100)
        
        if risk_score_val < 30:
            risk_level, risk_color, risk_icon = "LOW RISK", "#10b981", "🟢"
            recommendations = "✅ Normal conditions. Standard safety protocols sufficient."
        elif risk_score_val < 60:
            risk_level, risk_color, risk_icon = "MEDIUM RISK", "#f59e0b", "🟡"
            recommendations = "⚠️ Exercise caution. Reduce speed and stay alert."
        else:
            risk_level, risk_color, risk_icon = "HIGH RISK", "#ef4444", "🔴"
            recommendations = "🚨 Critical risk! Avoid travel if possible. Deploy emergency protocols."

        severity_labels = {0: 'Minor', 1: 'Major', 2: 'Fatal'}
        
        result = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                    padding: 20px; background: #ffffff; border-radius: 12px; 
                    border: 1px solid #e5e7eb; margin-top: 10px;">
            
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <div style="font-size: 40px; margin-right: 16px;">{risk_icon}</div>
                <div>
                    <div style="font-size: 24px; font-weight: 700; color: {risk_color};">
                        {risk_level}
                    </div>
                    <div style="font-size: 14px; color: #6b7280; margin-top: 4px;">
                        Risk Score: {risk_score_val:.1f}/100 | Predicted: {severity_labels[pred]} Accident
                    </div>
                </div>
            </div>
            
            <div style="margin-bottom: 20px;">
                <div style="margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px; color: #374151;">
                        <span>Minor Severity</span>
                        <span style="font-weight: 600;">{proba[0]*100:.1f}%</span>
                    </div>
                    <div style="background: #e5e7eb; height: 8px; border-radius: 4px;">
                        <div style="background: #10b981; width: {proba[0]*100}%; height: 8px; border-radius: 4px;"></div>
                    </div>
                </div>
                <div style="margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px; color: #374151;">
                        <span>Major Severity</span>
                        <span style="font-weight: 600;">{proba[1]*100:.1f}%</span>
                    </div>
                    <div style="background: #e5e7eb; height: 8px; border-radius: 4px;">
                        <div style="background: #f59e0b; width: {proba[1]*100}%; height: 8px; border-radius: 4px;"></div>
                    </div>
                </div>
                <div style="margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px; color: #374151;">
                        <span>Fatal Severity</span>
                        <span style="font-weight: 600;">{proba[2]*100:.1f}%</span>
                    </div>
                    <div style="background: #e5e7eb; height: 8px; border-radius: 4px;">
                        <div style="background: #ef4444; width: {proba[2]*100}%; height: 8px; border-radius: 4px;"></div>
                    </div>
                </div>
            </div>
            
            <div style="padding: 16px; background: #f9fafb; border-radius: 8px; 
                        border-left: 4px solid {risk_color}; color: #374151;">
                <strong>📋 Recommendation:</strong> {recommendations}
            </div>
            
            <div style="margin-top: 16px; font-size: 12px; color: #9ca3af; text-align: right;">
                Model: {model_choice} | Confidence: {max(proba)*100:.1f}%
            </div>
        </div>
        """
        
        return result
        
    except Exception as e:
        return f"""
        <div style="padding: 20px; background: #fef2f2; border-radius: 12px; 
                    border: 1px solid #fecaca; color: #dc2626;">
            <strong>Error:</strong> {str(e)}
        </div>
        """

# ============================================
# 5. CHART FUNCTIONS
# ============================================

def create_risk_by_hour_chart():
    """Hourly risk chart"""
    hours = list(stats['hourly_risk'].keys())
    risks = list(stats['hourly_risk'].values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=risks,
        mode='lines+markers',
        line=dict(color='#ef4444', width=2),
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.1)'
    ))
    
    fig.update_layout(
        title='Risk Level by Hour of Day',
        xaxis_title='Hour',
        yaxis_title='Average Risk Score',
        template='plotly_white',
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=False
    )
    
    return fig

def create_city_risk_chart():
    """City risk chart"""
    cities = list(stats['city_risk'].keys())[:10]
    risks = list(stats['city_risk'].values())[:10]
    
    fig = go.Figure(data=[
        go.Bar(
            x=cities, y=risks,
            marker_color='#3b82f6',
            text=[f'{r:.3f}' for r in risks],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Top 10 Cities by Risk Score',
        xaxis_title='City',
        yaxis_title='Average Risk Score',
        template='plotly_white',
        height=350,
        margin=dict(l=40, r=20, t=50, b=80)
    )
    
    return fig

def create_weather_impact_chart():
    """Weather impact chart"""
    weathers = list(stats['weather_risk'].keys())
    risks = list(stats['weather_risk'].values())
    colors = {'Clear': '#10b981', 'Rain': '#3b82f6', 'Fog': '#6b7280'}
    
    fig = go.Figure(data=[
        go.Bar(
            x=weathers, y=risks,
            marker_color=[colors.get(w, '#ef4444') for w in weathers],
            text=[f'{r:.3f}' for r in risks],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Risk Level by Weather Condition',
        xaxis_title='Weather',
        yaxis_title='Average Risk Score',
        template='plotly_white',
        height=350,
        margin=dict(l=40, r=20, t=50, b=60)
    )
    
    return fig

def create_model_comparison_chart():
    """Model performance comparison"""
    models = ['Random Forest', 'XGBoost', 'KNN', 'Logistic Regression']
    accuracy = [0.692, 0.656, 0.527, 0.497]
    precision = [0.71, 0.67, 0.54, 0.51]
    recall = [0.69, 0.66, 0.53, 0.50]
    
    fig = go.Figure(data=[
        go.Bar(name='Accuracy', x=models, y=accuracy),
        go.Bar(name='Precision', x=models, y=precision),
        go.Bar(name='Recall', x=models, y=recall)
    ])
    
    fig.update_layout(
        title='Model Performance Comparison',
        barmode='group',
        template='plotly_white',
        height=400,
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def get_city_ranking_table():
    """City risk ranking dataframe"""
    city_stats = df.groupby('city').agg({
        'risk_score': 'mean',
        'city': 'count',
        'accident_severity': lambda x: (x == 'fatal').sum()
    }).rename(columns={'city': 'total_accidents'})
    
    city_stats.columns = ['Avg Risk', 'Total Accidents', 'Fatal Accidents']
    city_stats['Fatal Rate (%)'] = (city_stats['Fatal Accidents'] / city_stats['Total Accidents'] * 100).round(2)
    city_stats = city_stats.sort_values('Avg Risk', ascending=False)
    
    return city_stats.reset_index()
# Pre-compute all charts at startup
print("  Pre-computing charts...")
_cached_hourly_chart = None
_cached_city_chart = None
_cached_weather_chart = None
_cached_model_chart = None
_cached_city_table = None

def get_hourly_chart():
    global _cached_hourly_chart
    if _cached_hourly_chart is None:
        _cached_hourly_chart = create_risk_by_hour_chart()
    return _cached_hourly_chart

def get_city_chart():
    global _cached_city_chart
    if _cached_city_chart is None:
        _cached_city_chart = create_city_risk_chart()
    return _cached_city_chart

def get_weather_chart():
    global _cached_weather_chart
    if _cached_weather_chart is None:
        _cached_weather_chart = create_weather_impact_chart()
    return _cached_weather_chart

def get_model_chart():
    global _cached_model_chart
    if _cached_model_chart is None:
        _cached_model_chart = create_model_comparison_chart()
    return _cached_model_chart

def get_city_table():
    global _cached_city_table
    if _cached_city_table is None:
        _cached_city_table = get_city_ranking_table()
    return _cached_city_table

# Pre-load all charts
get_hourly_chart()
get_city_chart()
get_weather_chart()
get_model_chart()
get_city_table()
print("  Charts pre-computed!")

# ============================================
# 6. CSS
# ============================================

css = """
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body, .gradio-container {
    background: #f3f4f6 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif !important;
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

.gradio-container {
    max-width: 100% !important;
    padding: 16px !important;
}

.main-header {
    background: #ffffff;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.stat-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 20px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

.tabs, .tab-nav {
    background: #ffffff !important;
    border-radius: 10px !important;
    padding: 6px !important;
    border: 1px solid #e5e7eb !important;
    margin-bottom: 16px !important;
}

.tab-nav button {
    color: #374151 !important;
    background: transparent !important;
    border: none !important;
    padding: 10px 20px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    transition: all 0.15s !important;
    margin: 2px !important;
}

.tab-nav button:hover {
    background: #f3f4f6 !important;
    color: #111827 !important;
}

.tab-nav button.selected {
    background: #3b82f6 !important;
    color: #ffffff !important;
}

.gr-box, .gr-form, .gr-panel {
    background: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid #e5e7eb !important;
}

label, .label-text, span, p, div {
    color: #374151 !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #111827 !important;
}

input, select, textarea, .gr-dropdown, .gr-slider {
    background: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
}

input:focus, select:focus, textarea:focus {
    border-color: #3b82f6 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
}

.gr-button-primary, button[variant="primary"] {
    background: #3b82f6 !important;
    color: #ffffff !important;
    border: none !important;
    padding: 10px 24px !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
}

.gr-button-primary:hover {
    background: #2563eb !important;
}

.gr-button-secondary {
    background: #ffffff !important;
    color: #374151 !important;
    border: 1px solid #d1d5db !important;
    padding: 10px 24px !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
}

.gr-button-secondary:hover {
    background: #f9fafb !important;
}

.js-plotly-plot {
    background: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid #e5e7eb !important;
    padding: 12px !important;
}

.gr-dataframe {
    background: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid #e5e7eb !important;
}

.gr-dataframe table {
    color: #374151 !important;
}

.gr-dataframe th {
    background: #f9fafb !important;
    color: #111827 !important;
    font-weight: 600 !important;
}

.gr-row, .gr-column {
    width: 100% !important;
}

footer {
    display: none !important;
}
"""

# ============================================
# 7. CHAT WIDGET
# ============================================

CHAT_WIDGET = """
<div id="fab" onclick="toggleChat()"
  style="position:fixed;bottom:24px;right:24px;width:56px;height:56px;
  background:#3b82f6;border-radius:50%;display:flex;align-items:center;
  justify-content:center;cursor:pointer;z-index:9999;
  box-shadow:0 4px 12px rgba(59,130,246,0.4);font-size:26px;
  border:2px solid white;">💬</div>

<div id="cwin" style="display:none;position:fixed;bottom:92px;right:24px;
  width:360px;background:white;border-radius:12px;z-index:9999;
  box-shadow:0 4px 20px rgba(0,0,0,0.15);flex-direction:column;
  overflow:hidden;border:1px solid #e5e7eb">

  <div style="background:#3b82f6;padding:14px 18px;color:white;
    font-weight:600;font-size:15px;display:flex;
    justify-content:space-between;align-items:center">
    <span>🚗 Road Safety AI</span>
    <span onclick="toggleChat()"
      style="cursor:pointer;font-size:22px;line-height:1;color:white">×</span>
  </div>

  <div id="cmsgs" style="height:320px;overflow-y:auto;padding:14px;
    display:flex;flex-direction:column;gap:10px;background:#f9fafb">
    <div style="background:#3b82f6;color:white;padding:10px 14px;
      border-radius:14px 14px 14px 2px;font-size:13px;
      max-width:88%;line-height:1.5">
      Hi! Ask me anything about road accidents, risk factors, or this project!
    </div>
  </div>

  <div style="padding:12px;display:flex;gap:8px;
    border-top:1px solid #e5e7eb;background:white">
    <input id="cinput" type="text" placeholder="Type your message..."
      style="flex:1;padding:10px 14px;border:1px solid #d1d5db;
      border-radius:20px;font-size:13px;outline:none;color:#111827;
      background:white"
      onkeydown="if(event.key==='Enter')sendMsg()"/>
    <button onclick="sendMsg()"
      style="background:#3b82f6;color:white;border:none;border-radius:50%;
      width:40px;height:40px;cursor:pointer;font-size:16px;
      display:flex;align-items:center;justify-content:center">➤</button>
  </div>
</div>

<script>
function toggleChat(){
  var w = document.getElementById('cwin');
  w.style.display = (w.style.display === 'flex') ? 'none' : 'flex';
  if(w.style.display === 'flex'){
    setTimeout(function(){ document.getElementById('cinput').focus(); }, 100);
  }
}

async function sendMsg(){
  var inp = document.getElementById('cinput');
  var msg = inp.value.trim();
  if(!msg) return;
  inp.value = '';
  var box = document.getElementById('cmsgs');

  var u = document.createElement('div');
  u.style.cssText = 'background:#10b981;color:white;padding:10px 14px;' +
    'border-radius:14px 14px 2px 14px;font-size:13px;max-width:88%;' +
    'align-self:flex-end;margin-left:auto;line-height:1.5';
  u.textContent = msg;
  box.appendChild(u);

  var t = document.createElement('div');
  t.id = 'typing';
  t.style.cssText = 'background:#e5e7eb;color:#6b7280;padding:10px 14px;' +
    'border-radius:14px 14px 14px 2px;font-size:13px;max-width:60%';
  t.textContent = 'Typing...';
  box.appendChild(t);
  box.scrollTop = box.scrollHeight;

  try {
    const r = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: msg})
    });
    const d = await r.json();
    var typing = document.getElementById('typing');
    if(typing) typing.remove();

    var b = document.createElement('div');
    b.style.cssText = 'background:white;color:#374151;padding:10px 14px;' +
      'border-radius:14px 14px 14px 2px;font-size:13px;max-width:88%;' +
      'border:1px solid #e5e7eb;line-height:1.5';
    b.textContent = d.response;
    box.appendChild(b);
  } catch(e) {
    var typing = document.getElementById('typing');
    if(typing) typing.remove();
    var er = document.createElement('div');
    er.style.cssText = 'background:#fef2f2;color:#dc2626;padding:10px 14px;' +
      'border-radius:14px;font-size:13px;border:1px solid #fecaca';
    er.textContent = 'Sorry, something went wrong.';
    box.appendChild(er);
  }
  box.scrollTop = box.scrollHeight;
}
</script>
"""

# ============================================
# 8. LOAD MAP
# ============================================

try:
    with open('models/hotspot_map.html', 'r', encoding='utf-8') as f:
        map_html = f.read()
    map_encoded = map_html.replace('"', '&quot;').replace("'", "&#39;")
except:
    map_encoded = "<h3 style='padding:40px;text-align:center;'>Map not found. Please run map_generator.py first.</h3>"

# ============================================
# 9. BUILD UI
# ============================================

with gr.Blocks(title="Indian Road Accident Risk Analyzer", css=css) as demo:
    
    # Header
    total_accidents = len(df)
    fatal_count = (df['accident_severity'] == 'fatal').sum()
    fatal_rate = (fatal_count / total_accidents * 100)
    
    gr.HTML(f"""
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h1 style="font-size: 28px; font-weight: 700; color: #111827; margin-bottom: 8px;">
                    🚗 Indian Road Accident Risk Analyzer
                </h1>
                <p style="color: #6b7280; font-size: 14px;">
                    ML-Powered Predictive Analytics • Real-time Risk Assessment
                </p>
            </div>
            <div style="text-align: right;">
                <div style="display: flex; gap: 20px;">
                    <div>
                        <div style="font-size: 24px; font-weight: 700; color: #111827;">
                            {total_accidents:,}
                        </div>
                        <div style="font-size: 12px; color: #6b7280;">Total Records</div>
                    </div>
                    <div>
                        <div style="font-size: 24px; font-weight: 700; color: #ef4444;">
                            {fatal_rate:.1f}%
                        </div>
                        <div style="font-size: 12px; color: #6b7280;">Fatal Rate</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """)
    
    # Tabs
    with gr.Tabs():
        
        # TAB 1: DASHBOARD
        with gr.Tab("📊 Dashboard"):
            with gr.Row():
                avg_risk = df['risk_score'].mean()
                fatal_count_val = (df['accident_severity'] == 'fatal').sum()
                
                gr.HTML(f"""
                <div style="display: flex; gap: 16px; width: 100%;">
                    <div class="stat-card" style="flex: 1;">
                        <div style="font-size: 13px; color: #6b7280; margin-bottom: 8px;">Average Risk Score</div>
                        <div style="font-size: 32px; font-weight: 700; color: #f59e0b;">{avg_risk:.3f}</div>
                    </div>
                    <div class="stat-card" style="flex: 1;">
                        <div style="font-size: 13px; color: #6b7280; margin-bottom: 8px;">Fatal Accidents</div>
                        <div style="font-size: 32px; font-weight: 700; color: #ef4444;">{fatal_count_val:,}</div>
                    </div>
                    <div class="stat-card" style="flex: 1;">
                        <div style="font-size: 13px; color: #6b7280; margin-bottom: 8px;">Cities Covered</div>
                        <div style="font-size: 32px; font-weight: 700; color: #3b82f6;">8</div>
                    </div>
                    <div class="stat-card" style="flex: 1;">
                        <div style="font-size: 13px; color: #6b7280; margin-bottom: 8px;">Model Accuracy</div>
                        <div style="font-size: 32px; font-weight: 700; color: #10b981;">69.2%</div>
                    </div>
                </div>
                """)
            
            with gr.Row():
                with gr.Column():
                    gr.Plot(value=get_hourly_chart(), label="Hourly Risk Pattern")
                with gr.Column():
                    gr.Plot(value=get_city_chart(), label="City Risk Ranking")
            
            with gr.Row():
                with gr.Column():
                    gr.Plot(value=get_weather_chart(), label="Weather Impact")
                with gr.Column():
                    city_table = gr.Dataframe(value=get_city_table(), label="City Risk Details")
        
        # TAB 2: PREDICT RISK
        with gr.Tab("🔮 Predict Risk"):
            with gr.Row():
                with gr.Column():
                    city = gr.Dropdown(list(city_map.keys()), label="City", value="Mumbai")
                    weather = gr.Dropdown(list(weather_map.keys()), label="Weather", value="Clear")
                    visibility = gr.Dropdown(list(vis_map.keys()), label="Visibility", value="High")
                    road_type = gr.Dropdown(list(road_map.keys()), label="Road Type", value="Urban")
                    traffic_density = gr.Dropdown(list(traffic_map.keys()), label="Traffic Density", value="Medium")
                
                with gr.Column():
                    hour = gr.Slider(0, 23, value=18, step=1, label="Hour of Day")
                    day_of_week = gr.Dropdown(list(day_map.keys()), label="Day of Week", value="Friday")
                    temperature = gr.Slider(5, 50, value=28, step=1, label="Temperature (°C)")
                    vehicles_involved = gr.Slider(1, 10, value=2, step=1, label="Vehicles Involved")
                    casualties = gr.Slider(0, 10, value=1, step=1, label="Casualties")
                
                with gr.Column():
                    festival = gr.Dropdown(list(festival_map.keys()), label="Festival", value="None")
                    cause = gr.Dropdown(list(cause_map.keys()), label="Primary Cause", value="Overspeeding")
                    lanes = gr.Slider(1, 6, value=2, step=1, label="Number of Lanes")
                    risk_score = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Initial Risk Score")
                    
                    with gr.Row():
                        is_weekend = gr.Checkbox(label="Weekend", value=False)
                        is_peak_hour = gr.Checkbox(label="Peak Hour", value=True)
                        traffic_signal = gr.Checkbox(label="Traffic Signal", value=True)
            
            model_choice = gr.Radio(['Random Forest', 'XGBoost', 'KNN', 'Logistic Regression'], 
                                   label="Select Model", value="Random Forest")
            
            predict_btn = gr.Button("🔮 Analyze Risk", variant="primary")
            prediction_output = gr.HTML()
            
            predict_btn.click(
                fn=predict_risk,
                inputs=[city, weather, road_type, visibility, traffic_density,
                       hour, is_weekend, is_peak_hour, temperature, lanes,
                       traffic_signal, vehicles_involved, casualties,
                       risk_score, day_of_week, festival, cause, model_choice],
                outputs=prediction_output
            )
        
        # TAB 3: HOTSPOT MAP
        with gr.Tab("🗺️ Hotspot Map"):
            gr.HTML(f"""
            <div style="background: white; border-radius: 10px; border: 1px solid #e5e7eb; overflow: hidden;">
                <iframe srcdoc="{map_encoded}" width="100%" height="600" 
                        style="border: none; display: block;">
                </iframe>
            </div>
            """)
        
        # TAB 4: MODEL PERFORMANCE
        with gr.Tab("📈 Model Performance"):
            gr.Plot(value=get_model_chart())
            
            gr.Markdown("""
            ### Model Performance Summary
            
            | Model | Accuracy | Precision | Recall | F1-Score |
            |-------|----------|-----------|--------|----------|
            | Random Forest | 69.2% | 71.0% | 69.0% | 70.0% |
            | XGBoost | 65.6% | 67.0% | 66.0% | 66.0% |
            | KNN | 52.7% | 54.0% | 53.0% | 53.0% |
            | Logistic Regression | 49.7% | 51.0% | 50.0% | 50.0% |
            
            ### Key Insights
            - **Random Forest** performs best overall with 69.2% accuracy
            - **Fatal accidents** are detected with high recall (99%)
            - **Major severity** is the most challenging to predict
            - **Risk Score** is the most important predictive feature
            """)
    
    # Chat Widget (only if Gemini is available)
    if GEMINI_AVAILABLE:
        gr.HTML(CHAT_WIDGET)

# ============================================
# 10. CHAT API
# ============================================

app_api = demo.app

@app_api.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    message = data.get("message", "")
    response = chat_with_gemini(message)
    return JSONResponse({"response": response})

# ============================================
# 11. LAUNCH
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 INDIAN ROAD ACCIDENT RISK ANALYZER")
    print("=" * 60)
    print(f"\n✅ System Ready!")
    print(f"📍 Access the dashboard at: http://localhost:7860")
    print(f"\n📊 Dataset Statistics:")
    print(f"  • Total Records: {stats['total_accidents']:,}")
    print(f"  • Fatal Accidents: {stats['fatal_count']:,}")
    print(f"  • Average Risk Score: {stats['avg_risk']:.3f}")
    print("\n" + "=" * 60)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )