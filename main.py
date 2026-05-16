# Author: Xandre Arsoler
# Student ID: TUPM-25-0260
# Project: VIB-01 Natural Frequency Resonance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import skew

# -------------------------------
# Module 1: Data Ingestion
# -------------------------------
def load_data(path):
    try:
        df = pd.read_csv(path)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# -------------------------------
# Module 2: Cleaning
# -------------------------------
def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    return df

# -------------------------------
# Module 3: Analysis
# -------------------------------
def descriptive_stats(df, column):
    stats = {
        "mean": np.mean(df[column]),
        "median": np.median(df[column]),
        "std_dev": np.std(df[column]),
        "variance": np.var(df[column]),
        "skewness": skew(df[column])
    }
    return stats

def correlation_matrix(df):
    return df.corr()

# -------------------------------
# Module 4: Visualization
# -------------------------------
def static_visualizations(df, column):
    plt.figure(figsize=(6,4))
    sns.histplot(df[column], kde=True)
    plt.title("Histogram of " + column)
    plt.savefig("outputs/static_histogram.png")

    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[column])
    plt.title("Boxplot of " + column)
    plt.savefig("outputs/static_boxplot.png")

    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df.index, y=df[column])
    plt.title("Scatter Plot of " + column)
    plt.savefig("outputs/static_scatter.png")

    plt.figure(figsize=(6,4))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("outputs/heatmap.png")

def animated_visualizations(df, column):
    fig = px.histogram(df, x=column, animation_frame="UDI", nbins=30, title="Distribution Shift Over Time")
    fig.write_html("outputs/animation_distribution.html")

    fig2 = px.scatter(df, x="UDI", y=column, animation_frame="UDI", title="Trend Animation")
    fig2.write_html("outputs/animation_trend.html")

# -------------------------------
# Module 5: Pipeline Execution
# -------------------------------
def run_pipeline():
    df = load_data("data/dataset_original.csv")
    df = clean_data(df)
    df.to_csv("data/dataset_cleaned.csv", index=False)

    stats = descriptive_stats(df, "Air temperature [K]")
    print("Descriptive Stats:", stats)

    corr = correlation_matrix(df)
    print("Correlation Matrix:\n", corr)

    static_visualizations(df, "Air temperature [K]")
    animated_visualizations(df, "Air temperature [K]")

if __name__ == "__main__":
    run_pipeline()
