# Author: Xandre Arsoler
# Student ID: TUPM-25-0260
# Project: VIB-01 Natural Frequency Resonance

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
import matplotlib.animation as animation

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# -------------------------------
# Module 1: Data Ingestion
# -------------------------------
def load_data(path):
    try:
        df = pd.read_csv(path)
        print(f"Data loaded successfully. Found {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# -------------------------------
# Module 2: Cleaning (FORCED TO NOT DROP ROWS)
# -------------------------------
def clean_data(df):
    # Just drop exact duplicates, do NOT drop rows with NaN values
    df = df.drop_duplicates()
    
    # Ensure our target column is numeric, convert errors to numbers smoothly
    target_col = "Air temperature [K]"
    if target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        # Fill any accidental missing values with the average so no rows are deleted
        df[target_col] = df[target_col].fillna(df[target_col].mean())
        
    return df

# -------------------------------
# Module 3: Analysis
# -------------------------------
def descriptive_stats(df, column):
    stats = {
        "mean": np.nanmean(df[column]),
        "median": np.nanmedian(df[column]),
        "std_dev": np.nanstd(df[column]),
        "variance": np.nanvar(df[column]),
        "skewness": skew(df[column], nan_policy='omit') if len(df) > 0 else 0
    }
    return stats

def correlation_matrix(df):
    return df.corr()

# -------------------------------
# Module 4: Visualization
# -------------------------------
def static_visualizations(df, column):
    plt.figure(figsize=(6,4))
    sns.histplot(df[column].dropna(), kde=True)
    plt.title("Histogram of " + column)
    plt.savefig("outputs/static_histogram.png")
    plt.close()

    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[column])
    plt.title("Boxplot of " + column)
    plt.savefig("outputs/static_boxplot.png")
    plt.close()

    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df.index, y=df[column])
    plt.title("Scatter Plot of " + column)
    plt.savefig("outputs/static_scatter.png")
    plt.close()

    plt.figure(figsize=(6,4))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("outputs/heatmap.png")
    plt.close()


def animated_visualizations(df, column):
    # Safety Check: If data has 0 rows, don't try to build a GIF
    if len(df) == 0:
        print("Skipping animations: Dataset is empty.")
        return

    # 1. Distribution Shift GIF
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    bins = np.linspace(df[column].min(), df[column].max(), 30)
    
    def update_hist(frame):
        ax1.clear()
        current_data = df.iloc[:frame + 1][column]
        ax1.hist(current_data, bins=bins, color='skyblue', edgecolor='black')
        ax1.set_title(f"Distribution Shift (Frame {frame})")
        ax1.set_xlabel(column)
        ax1.set_ylabel("Frequency")
        ax1.set_ylim(0, max(10, len(df) // 4)) 

    # Break frames down into 20 chunks so it compiles quickly
    step = max(1, len(df) // 20)
    frames_hist = list(range(0, len(df), step))

    ani1 = animation.FuncAnimation(fig1, update_hist, frames=frames_hist, interval=200)
    ani1.save("outputs/animation_distribution.gif", writer="pillow")
    plt.close(fig1)

    # 2. Trend Animation GIF
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    x = df["UDI"] if "UDI" in df.columns else df.index
    y = df[column]
    line, = ax2.plot([], [], color="orange", lw=2)
    
    ax2.set_xlim(x.min(), x.max())
    ax2.set_ylim(y.min(), y.max())
    ax2.set_title("Trend Animation")
    ax2.set_xlabel("UDI" if "UDI" in df.columns else "Index")
    ax2.set_ylabel(column)

    def init_trend():
        line.set_data([], [])
        return line,

    def update_trend(frame):
        line.set_data(x[:frame], y[:frame])
        return line,

    ani2 = animation.FuncAnimation(
        fig2, update_trend, frames=frames_hist,
        init_func=init_trend, interval=100, blit=False
    )
    ani2.save("outputs/animation_trend.gif", writer="pillow")
    plt.close(fig2)


def gif_animation(df, column):
    if len(df) == 0: return
    
    fig, ax = plt.subplots(figsize=(8,5))
    x = df.index
    y = df[column]
    line, = ax.plot([], [], lw=2, color="royalblue")

    ax.set_xlim(0, len(x))
    ax.set_ylim(min(y), max(y))
    ax.set_title(f"GIF Animation of {column}")
    ax.set_xlabel("Index")
    ax.set_ylabel(column)

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        line.set_data(x[:frame], y[:frame])
        return line,

    step = max(1, len(df) // 40)
    frames_gif = list(range(0, len(x), step))

    ani = animation.FuncAnimation(
        fig, update, frames=frames_gif,
        init_func=init, interval=50, blit=False
    )
    ani.save("outputs/animated_trend.gif", writer="pillow")
    plt.close(fig)

# -------------------------------
# Module 5: Pipeline Execution
# -------------------------------
def run_pipeline():
    df = load_data("data/dataset_original.csv")
    if df is None:
        return
        
    df = clean_data(df)
    target_col = "Air temperature [K]"
    
    if target_col not in df.columns:
        # Fallback to the first numeric column if named column doesn't match
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            target_col = numeric_cols[0]
        else:
            print("Error: No numeric columns found in your CSV file.")
            return

    df.to_csv("data/dataset_cleaned.csv", index=False)

    stats = descriptive_stats(df, target_col)
    print("Descriptive Stats:", stats)

    numeric_df = df.select_dtypes(include=[np.number])
    corr = correlation_matrix(numeric_df)
    print("Correlation Matrix Calculated Successfully.")

    print("Generating static visuals...")
    static_visualizations(df, target_col)
    
    print("Generating animated GIFs...")
    animated_visualizations(df, target_col)
    gif_animation(df, target_col)
    print("Pipeline completed successfully! Check your 'outputs' directory.")

if __name__ == "__main__":
    run_pipeline()