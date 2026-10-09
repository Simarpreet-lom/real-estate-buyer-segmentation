

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Real Estate Buyer Segmentation",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 Real Estate Buyer Segmentation")
st.subheader("Investment Profiling and Market Intelligence")

base_dir = Path(__file__).parent
clients_path = base_dir / "clients.csv"
properties_path = base_dir / "properties.csv"

if not clients_path.exists() or not properties_path.exists():
    st.error("Please keep clients.csv and properties.csv in the same folder as app.py.")
    st.stop()

clients = pd.read_csv(clients_path)
properties = pd.read_csv(properties_path)

st.success("Client and property datasets loaded successfully.")

st.header("Project Overview")
col1, col2 = st.columns(2)
col1.metric("Total Clients", len(clients))
col2.metric("Total Properties", len(properties))

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a section",
    ["Buyer Segmentation", "Investment Profiling", "Market Insights"]
)
  
if page == "Buyer Segmentation":
    import numpy as np

    st.header("Buyer Segmentation Using K-Means")
    candidate_features = [
        "client_type",
        "acquisition_purpose",
        "satisfaction_score",
        "loan_applied",
        "country",
        "region",
        "referral_channel"
    ]

    features = [c for c in candidate_features if c in clients.columns]

    if len(features) < 2:
        st.error("Not enough buyer features found in the dataset.")
        st.write("Available columns:", clients.columns.tolist())
        st.stop()

    data = clients[features].copy()
    numeric_features = data.select_dtypes(include="number").columns.tolist()

    for col in data.columns:
        if col not in numeric_features:
            data[col] = data[col].fillna("Unknown").astype(str)

    for col in numeric_features:
        data[col] = data[col].fillna(data[col].median())

    encoded = pd.get_dummies(data, dtype=float)
    X = encoded.to_numpy(dtype=float)

    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1
    X = (X - means) / stds

    max_clusters = min(6, len(clients) - 1)

    if max_clusters < 2:
        st.error("At least three client records are recommended.")
        st.stop()

    k = st.slider(
        "Choose number of buyer segments",
        min_value=2,
        max_value=max_clusters,
        value=min(3, max_clusters)
    )

    rng = np.random.default_rng(42)
    centers = X[rng.choice(len(X), size=k, replace=False)].copy()

    for _ in range(100):
        distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)

        new_centers = centers.copy()
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                new_centers[j] = members.mean(axis=0)

        if np.allclose(centers, new_centers):
            break

        centers = new_centers

    distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = distances.argmin(axis=1)

    results = clients.copy()
    results["Buyer_Segment"] = labels + 1

    st.success("Buyer segmentation completed.")

    st.subheader("Segmentation Results")
    st.dataframe(results, use_container_width=True)

    st.subheader("Buyers in Each Segment")
    counts = results["Buyer_Segment"].value_counts().sort_index()
    st.bar_chart(counts)

    st.subheader("Segment Profiles")
    for segment in sorted(results["Buyer_Segment"].unique()):
        group = results[results["Buyer_Segment"] == segment]
        with st.expander(f"Buyer Segment {segment} — {len(group)} clients"):
            st.dataframe(group[features].head(20), use_container_width=True)

    st.download_button(
        "Download Segmentation Results (CSV)",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="buyer_segmentation_results.csv",
        mime="text/csv"
    )
elif page == "Investment Profiling":
    st.header("Investment Profiling")
    st.write("Explore property values, sizes, and listing status.")

    if "sale_price" in properties.columns:
        prices = pd.to_numeric(
            properties["sale_price"].astype(str).str.replace(
                r"[$,]", "", regex=True
            ),
            errors="coerce"
        )
        st.metric(
            "Average Sale Price",
            f"${prices.mean():,.2f}" if prices.notna().any() else "Unavailable"
        )

    if "unit_category" in properties.columns:
        st.subheader("Properties by Unit Category")
        st.bar_chart(properties["unit_category"].value_counts())

    if "listing_status" in properties.columns:
        st.subheader("Listing Status")
        st.bar_chart(properties["listing_status"].value_counts())

    st.subheader("Property Data Preview")
    st.dataframe(properties.head(20), use_container_width=True)

else:
    st.header("Market Insights")

    if "acquisition_purpose" in clients.columns:
        st.subheader("Buyer Acquisition Purpose")
        st.bar_chart(clients["acquisition_purpose"].value_counts())

    if "country" in clients.columns:
        st.subheader("Buyers by Country")
        st.bar_chart(clients["country"].value_counts().head(10))

    if "referral_channel" in clients.columns:
        st.subheader("Referral Channels")
        st.bar_chart(clients["referral_channel"].value_counts())

    st.subheader("Client Data Preview")
    st.dataframe(clients.head(20), use_container_width=True)
