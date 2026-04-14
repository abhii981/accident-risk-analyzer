import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
import joblib
import os
from branca.colormap import LinearColormap

print("🗺️ Generating Enhanced Hotspot Map...")

# Load data
df = pd.read_csv('data/indian_roads_dataset.csv')
df['festival'] = df['festival'].fillna('none')

# Load clustered data
clustered = joblib.load('models/clustered_data.pkl')
df['cluster_kmeans'] = clustered['cluster_kmeans'].values
df['cluster_dbscan'] = clustered['cluster_dbscan'].values

# Create base map with multiple tile layers
m = folium.Map(
    location=[20.5937, 78.9629],
    zoom_start=5,
    tiles='CartoDB positron',
    control_scale=True
)

# Add different tile layers
folium.TileLayer('OpenStreetMap').add_to(m)
folium.TileLayer('CartoDB dark_matter').add_to(m)

# Create feature groups for better organization
heatmap_group = folium.FeatureGroup(name='🔥 Risk Heatmap')
cluster_group = folium.FeatureGroup(name='📍 Accident Clusters')
fatal_group = folium.FeatureGroup(name='⚠️ Fatal Accidents')
city_group = folium.FeatureGroup(name='🏙️ City Analysis')

# 1. Enhanced Heatmap with weighted risk
heat_data = []
for _, row in df.iterrows():
    # Weight by risk score
    weight = row['risk_score'] * 2
    if row['accident_severity'] == 'fatal':
        weight *= 1.5
    heat_data.append([row['latitude'], row['longitude'], weight])

HeatMap(
    heat_data,
    min_opacity=0.3,
    radius=15,
    blur=10,
    max_zoom=13,
    gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}
).add_to(heatmap_group)

# 2. Cluster visualization with enhanced styling
cluster_colors = {
    0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1', 
    3: '#96CEB4', 4: '#FFEAA7', 5: '#DDA0DD',
    6: '#98D8C8', 7: '#F7DC6F'
}

for cluster_id in sorted(df['cluster_kmeans'].unique()):
    cluster_df = df[df['cluster_kmeans'] == cluster_id]
    
    if len(cluster_df) < 5:  # Skip very small clusters
        continue
    
    center_lat = cluster_df['latitude'].mean()
    center_lon = cluster_df['longitude'].mean()
    count = len(cluster_df)
    avg_risk = cluster_df['risk_score'].mean()
    fatal_count = (cluster_df['accident_severity'] == 'fatal').sum()
    
    # Create detailed popup
    popup_html = f"""
    <div style="font-family: Arial, sans-serif; min-width: 200px;">
        <h4 style="margin: 0 0 10px; color: #333;">📍 Cluster {cluster_id}</h4>
        <table style="width: 100%; border-collapse: collapse;">
            <tr><td><b>Total Accidents:</b></td><td>{count}</td></tr>
            <tr><td><b>Fatal Accidents:</b></td><td>{fatal_count}</td></tr>
            <tr><td><b>Avg Risk Score:</b></td><td>{avg_risk:.3f}</td></tr>
            <tr><td><b>Risk Level:</b></td>
                <td style="color: {'red' if avg_risk > 0.6 else 'orange' if avg_risk > 0.4 else 'green'};">
                    {'HIGH' if avg_risk > 0.6 else 'MEDIUM' if avg_risk > 0.4 else 'LOW'}
                </td>
            </tr>
        </table>
    </div>
    """
    
    # Add cluster marker
    folium.CircleMarker(
        location=[center_lat, center_lon],
        radius=min(30, 10 + count/100),
        color=cluster_colors.get(cluster_id, '#808080'),
        fill=True,
        fillOpacity=0.6,
        weight=2,
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=f"Cluster {cluster_id}: {count} accidents, Risk: {avg_risk:.2f}"
    ).add_to(cluster_group)

# 3. Fatal accidents with detailed information
fatal_df = df[df['accident_severity'] == 'fatal'].sample(
    min(500, len(df[df['accident_severity'] == 'fatal'])), random_state=42
)
marker_cluster = MarkerCluster().add_to(fatal_group)

for _, row in fatal_df.iterrows():
    popup_html = f"""
    <div style="font-family: Arial, sans-serif;">
        <h4 style="margin: 0 0 8px; color: #d32f2f;">⚠️ Fatal Accident</h4>
        <b>City:</b> {row['city']}<br>
        <b>Weather:</b> {row['weather']}<br>
        <b>Road:</b> {row['road_type']}<br>
        <b>Risk Score:</b> {row['risk_score']:.2f}<br>
        <b>Time:</b> {row['time']}<br>
        <b>Cause:</b> {row['cause']}<br>
        <b>Casualties:</b> {row['casualties']}
    </div>
    """
    
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=8,
        color='#d32f2f',
        fill=True,
        fill_color='#ff5252',
        fill_opacity=0.8,
        weight=2,
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=f"Fatal Accident - {row['city']}"
    ).add_to(marker_cluster)

# 4. City center markers with statistics
city_stats = df.groupby('city').agg({
    'latitude': 'mean',
    'longitude': 'mean',
    'risk_score': 'mean',
    'accident_severity': lambda x: (x == 'fatal').sum(),
    'city': 'count'
}).rename(columns={'city': 'total_accidents'})

for city, row in city_stats.iterrows():
    popup_html = f"""
    <div style="font-family: Arial, sans-serif;">
        <h4 style="margin: 0 0 10px;">🏙️ {city}</h4>
        <b>Total Accidents:</b> {row['total_accidents']}<br>
        <b>Fatal Accidents:</b> {row['accident_severity']}<br>
        <b>Avg Risk Score:</b> {row['risk_score']:.3f}<br>
        <b>Risk Index:</b> {(row['risk_score'] * row['total_accidents']/100):.1f}
    </div>
    """
    
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=f"{city} - {row['total_accidents']} accidents",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(city_group)

# Add all feature groups to map
heatmap_group.add_to(m)
cluster_group.add_to(m)
fatal_group.add_to(m)
city_group.add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Add legend
colormap = LinearColormap(
    ['green', 'yellow', 'orange', 'red'],
    vmin=0, vmax=1,
    caption='Risk Level'
)
colormap.add_to(m)

# Add title with CSS
title_html = '''
<div style="position: fixed; 
            top: 10px; left: 50px; width: 350px; height: 70px; 
            background-color: white; border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            z-index: 9999; 
            padding: 12px;
            font-family: Arial, sans-serif;">
    <h3 style="margin: 0; color: #333;">🚗 Indian Road Accident Hotspots</h3>
    <p style="margin: 5px 0 0; color: #666; font-size: 12px;">
        Heatmap intensity shows risk concentration | Use layer control to toggle views
    </p>
</div>
'''
m.get_root().html.add_child(folium.Element(title_html))

# Save map
os.makedirs('models', exist_ok=True)
m.save('models/hotspot_map.html')

print("✅ Enhanced hotspot map generated successfully!")
print(f"📍 Map saved to: models/hotspot_map.html")
print(f"📊 Total clusters: {len(df['cluster_kmeans'].unique())}")
print(f"⚠️ Total fatal accidents: {len(fatal_df)}")