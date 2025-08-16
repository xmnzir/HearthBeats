# ============================================================
# File: geomap.py
# Author: Mohammed Munazir
# Description: Implements Geo 
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import requests
import folium
from streamlit_folium import st_folium



@st.cache_data(show_spinner=False)
def geocode_locations(locations):
    geolocator = Nominatim(user_agent="geo_topic_module")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    coords = {}
    for loc in locations:
        try:
            location = geocode(f"{loc}, UK")
            if location:
                coords[loc] = (location.latitude, location.longitude)
        except Exception as e:
            st.warning(f"Geocoding failed for {loc}: {e}")
            
    return coords


@st.cache_data(show_spinner=False)
def geocode_location(location):
    coords = geocode_locations([location])
    return coords.get(location, (None, None))




def run_geo_topic_module(df: pd.DataFrame):
    st.header("Geo-aware Topic, Sentiment & Emotion Analysis")

   
    df.columns = df.columns.str.strip()
    emotion_cols = ["happiness", "sadness", "anger", "fear", "surprise"]
    required_cols = ["Location", "Detected Topics", "Sentiment Score"] + emotion_cols

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"Missing required column(s): {missing_cols}")
        return

    
    unique_locations = df["Location"].dropna().unique()
    uk_coords = geocode_locations(unique_locations)
    df = df[df["Location"].isin(uk_coords.keys())].copy()
    df["lat"] = df["Location"].map(lambda x: uk_coords[x][0])
    df["lon"] = df["Location"].map(lambda x: uk_coords[x][1])

    
    st.subheader(" Sentiment Heatmap")
    sentiment_fig = px.density_mapbox(
        df,
        lat="lat",
        lon="lon",
        z="Sentiment Score",
        radius=25,
        hover_name="Location",
        color_continuous_scale="RdYlGn",
        mapbox_style="carto-positron",
        zoom=5,
        center={"lat": 54.5, "lon": -2.5},
    )
    st.plotly_chart(sentiment_fig)

    
    st.subheader("Emotion Heatmap")
    selected_emotion = st.selectbox("Choose an emotion:", emotion_cols)
    emo_fig = px.density_mapbox(
        df,
        lat="lat",
        lon="lon",
        z=selected_emotion,
        radius=25,
        hover_name="Location",
        color_continuous_scale="Plasma",
        mapbox_style="carto-positron",
        zoom=5,
        center={"lat": 54.5, "lon": -2.5},
    )
    st.plotly_chart(emo_fig)

    
    st.subheader("Geo-Topic Clusters")
    topic_text = df["Detected Topics"].astype(str).values
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(topic_text)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(tfidf_matrix.toarray())

    n_clusters = st.slider("Choose number of topic clusters", min_value=2, max_value=8, value=4)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(reduced)
    df["topic_cluster"] = kmeans.labels_

    cluster_fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        color="topic_cluster",
        hover_name="Detected Topics",
        zoom=5,
        mapbox_style="carto-positron",
        center={"lat": 54.5, "lon": -2.5},
    )
    st.plotly_chart(cluster_fig)

    
    st.subheader("Localised Insight Generator")
    selected_loc = st.selectbox("Select a location for detailed analysis:", sorted(df["Location"].unique()), key="Location")
    loc_df = df[df["Location"] == selected_loc]

    if loc_df.empty:
        st.warning("No data for selected location.")
        return

   
    top_topic = loc_df["Detected Topics"].value_counts().idxmax()
    avg_sentiment = loc_df["Sentiment Score"].mean()
    dominant_emotion = loc_df[emotion_cols].mean().idxmax()
    lowest_emotion = loc_df[emotion_cols].mean().idxmin()

    st.markdown(f"**Location:** `{selected_loc}`")
    st.markdown(f"**Most Frequent Topic:** `{top_topic}`")
    st.markdown(f"**Avg. Sentiment Score:** `{avg_sentiment:.2f}`")
    st.markdown(f"**Dominant Emotion:** `{dominant_emotion}`")
    st.markdown(f"**Least Expressed Emotion:** `{lowest_emotion}`")

    
    content_col = next((col for col in df.columns if col.lower() == "content"), None)
    if content_col:
        loc_df = loc_df.dropna(subset=[content_col])
        st.markdown("---")
        st.markdown("### Top Content by Emotion")
        if not loc_df.empty:
            emotion_scores = loc_df[emotion_cols].mean()
            default_emotion = emotion_scores.idxmax()
            emotion_filter = st.selectbox(
                "Filter top feedback by emotion:",
                emotion_cols,
                index=emotion_cols.index(default_emotion),
                help="See most intense emotional content."
            )

            top_emotion_df = loc_df.sort_values(emotion_filter, ascending=False).head(3)

            for idx, row in top_emotion_df.iterrows():
                st.markdown(f"#### Feedback {idx + 1}")
                st.markdown(f"> {row[content_col]}")
                st.markdown(f"- Emotion Strength ({emotion_filter}): `{row[emotion_filter]:.2f}`")
                st.markdown(f"- Topic: `{row['Detected Topics']}`")
                st.markdown(f"- Sentiment Score: `{row['Sentiment Score']:.2f}`")
                if 'Customer Idx' in row:
                    st.markdown(f"- Customer ID: `{row['Customer Idx']}`")
                if 'Date' in row:
                    st.markdown(f"- Date: `{row['Date']}`")
                st.markdown("---")

            
            st.markdown("### Top Positive Feedback")
            top_positive = loc_df.sort_values("Sentiment Score", ascending=False).iloc[0]
            st.markdown(f"> {top_positive[content_col]}")
            st.markdown(f"- Sentiment Score: `{top_positive['Sentiment Score']:.2f}`")
            st.markdown(f"- Topic: `{top_positive['Detected Topics']}`")

            
            st.markdown("### Top Negative Feedback")
            top_negative = loc_df.sort_values("Sentiment Score", ascending=True).iloc[0]
            st.markdown(f"> {top_negative[content_col]}")
            st.markdown(f"- Sentiment Score: `{top_negative['Sentiment Score']:.2f}`")
            st.markdown(f"- Topic: `{top_negative['Detected Topics']}`")
        else:
            st.info("No valid entries with content.")
    else:
        st.warning("No feedback (`Content`) column found in this dataset.")

    
    if "Customer Idx" in df.columns:
        issues = loc_df.groupby("Customer Idx")["Detected Topics"].apply(lambda x: ", ".join(x)).reset_index()
        st.markdown("**Customer-Level Issue Aggregation**")
        st.dataframe(issues, use_container_width=True)


    st.success(
        "This module enables **geo-personalised feedback aggregation**, empowering housing stakeholders "
        "to respond to community-level emotions, complaints, or praise with precision."
    )





@st.cache_data(show_spinner=False)
def get_nearby_amenities(lat, lon, radius=1000, place_types=None):
    """
    Fetch nearby amenities from Google Places API for multiple place types.

    Args:
        lat (float): Latitude.
        lon (float): Longitude.
        radius (int): Search radius in meters.
        place_types (list or None): List of place types (strings), e.g. ['school', 'park'].
                                    If None, fetch all types.

    Returns:
        List of dicts with keys: name, category, lat, lon.
    """
    GOOGLE_MAPS_API_KEY = "AIzaSyBfKyFf2mAS_c00VwL01w0vkpntsPLZlOg"  

    if place_types is None:
        place_types = ["school", "transit_station", "park", "store"]

    all_results = []

    for p_type in place_types:
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{lat},{lon}",
            "radius": radius,
            "type": p_type,
            "key": GOOGLE_MAPS_API_KEY,
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            places = [{
                "name": result["name"],
                "category": p_type,
                "lat": result["geometry"]["location"]["lat"],
                "lon": result["geometry"]["location"]["lng"]
            } for result in data.get("results", [])]

            all_results.extend(places)

        except Exception as e:
            st.warning(f"Google Places API request failed for type '{p_type}': {e}")

    
    seen = set()
    unique_results = []
    for place in all_results:
        identifier = (place["name"], place["lat"], place["lon"])
        if identifier not in seen:
            unique_results.append(place)
            seen.add(identifier)

    return unique_results


def run_nearby_infra_tab(df: pd.DataFrame):
    st.header("Nearby Infrastructure Lookup")
    st.markdown(
        "Find nearby points of interest (amenities, buildings, shops, tourism spots) around a selected location "
        "using Nominatim geocoding and Google Places API."
    )

    if "Location" not in df.columns:
        st.warning("No 'Location' column found in dataset.")
        return

    unique_locations = sorted(df["Location"].dropna().unique())
    selected_loc = st.selectbox("Select a location:", unique_locations, key="Infra")
    if not selected_loc:
        st.info("Please select a location.")
        return

    
    uk_coords = geocode_locations([selected_loc])
    if selected_loc not in uk_coords:
        st.warning(f"Could not geocode location: {selected_loc}")
        return
    lat, lon = uk_coords[selected_loc]

    categories = {
        "Education Institutions": "school",
        "Public Transport": "transit_station",
        "Parks": "park",
        "Retailers": "store"
    }

    selected_categories = st.multiselect(
        "Select infrastructure categories to search for:",
        options=list(categories.keys()),
        default=list(categories.keys())
    )
    if not selected_categories:
        st.info("Please select at least one infrastructure category.")
        return

    selected_types = [categories[cat] for cat in selected_categories]

    radius = st.slider("Search radius (meters):", 100, 5000, 1000, step=100)

    with st.spinner("Fetching nearby infrastructure..."):
        infra = get_nearby_amenities(lat, lon, radius, selected_types)

    if not infra:
        st.info("No nearby infrastructure found.")
        return

 
    m = folium.Map(location=[lat, lon], zoom_start=14,
                   tiles='https://mt1.google.com/vt/lyrs=r&x={x}&y={y}&z={z}',
                   attr='Google')

 
    folium.Marker(
        location=[lat, lon],
        popup=f"<b>{selected_loc}</b>",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)

   
    for place in infra:
        folium.Marker(
            location=[place['lat'], place['lon']],
            popup=f"{place['name']} ({place['category']})",
            icon=folium.Icon(color='blue', icon='glyphicon glyphicon-map-marker')
        ).add_to(m)

    st_folium(m, width=700, height=500)
   

    st.markdown(f"### Nearby Points of Interest near **{selected_loc}**")
    infra_df = pd.DataFrame(infra)
    st.dataframe(infra_df[["name", "category"]], use_container_width=True)





@st.cache_data(show_spinner=False)
def get_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
        "timezone": "auto",
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        st.error(f"OpenMeteo API error: {response.status_code}")
        return None
    return response.json()

def run_weather_tab(df: pd.DataFrame):
    st.header("Weather Metrics")

    if "Location" not in df.columns:
        st.warning("No 'Location' column found in dataset.")
        return

    unique_locations = sorted(df["Location"].dropna().unique())
    selected_loc = st.selectbox("Select a location:", unique_locations, key="weather")
    if not selected_loc:
        st.info("Please select a location.")
        return

    lat, lon = geocode_location(selected_loc)
    if lat is None or lon is None:
        st.warning(f"Could not geocode location: {selected_loc}")
        return

    with st.spinner("Fetching weather data..."):
        weather_data = get_weather(lat, lon)

    if weather_data is None:
        return

    current = weather_data.get("current_weather", {})
    daily = weather_data.get("daily", {})

    st.markdown(f"### Current Weather in {selected_loc}")
    st.write(f"- Temperature: {current.get('temperature')} °C")
    st.write(f"- Wind Speed: {current.get('windspeed')} km/h")
    st.write(f"- Weather Code: {current.get('weathercode')}")

    st.markdown("### 7-Day Forecast")
    df_forecast = pd.DataFrame({
        "Date": daily.get("time", []),
        "Max Temp (°C)": daily.get("temperature_2m_max", []),
        "Min Temp (°C)": daily.get("temperature_2m_min", []),
        "Precipitation (mm)": daily.get("precipitation_sum", []),
        "Weather Code": daily.get("weathercode", [])
    })

    st.dataframe(df_forecast, use_container_width=True)

