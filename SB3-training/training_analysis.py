import pandas as pd
import json
import os 

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser

def transform_csv(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file, comment='#', skipinitialspace=True)

    # Function to extract agent and value
    def extract_agent_value(s):
        parts = s.split('_')
        return f'pursuer_{parts[1]}', float(parts[2])

    # Apply the extraction to each column
    df['arousal_agent'], df['arousal_value'] = zip(*df['arousal'].apply(extract_agent_value))
    df['satiety_agent'], df['satiety_value'] = zip(*df['satiety'].apply(extract_agent_value))
    df['social_touch_agent'], df['social_touch_value'] = zip(*df['social-touch'].apply(extract_agent_value))

    # Reshape the dataframe
    result = pd.DataFrame({
        'agent': df['arousal_agent'],
        'r': df['r'],
        'l': df['l'],
        't': df['t'],
        'arousal': df['arousal_value'],
        'satiety': df['satiety_value'],
        'social_touch': df['social_touch_value']
    })

    # Sort the dataframe by agent and time
    result = result.sort_values(['agent', 't'])

    # Normalize arousal and satiety
    for agent in result['agent'].unique():
        mask = result['agent'] == agent
        result.loc[mask, 'arousal_norm'] = (result.loc[mask, 'arousal'] - result.loc[mask, 'arousal'].mean()) / result.loc[mask, 'arousal'].std()
        result.loc[mask, 'satiety_norm'] = (result.loc[mask, 'satiety'] - result.loc[mask, 'satiety'].mean()) / result.loc[mask, 'satiety'].std()

    # Save the result to a new CSV file
    result.to_csv(output_file, index=False)

    print(f"Transformed data saved to {output_file}")
    return result

def calculate_rolling_stats(df, window_size):
    stats = []
    for agent in df['agent'].unique():
        agent_data = df[df['agent'] == agent]
        arousal_mean = agent_data['arousal'].rolling(window=window_size).mean()
        arousal_std = agent_data['arousal'].rolling(window=window_size).std()
        satiety_mean = agent_data['satiety'].rolling(window=window_size).mean()
        satiety_std = agent_data['satiety'].rolling(window=window_size).std()
        arousal_norm_mean = agent_data['arousal_norm'].rolling(window=window_size).mean()
        arousal_norm_std = agent_data['arousal_norm'].rolling(window=window_size).std()
        satiety_norm_mean = agent_data['satiety_norm'].rolling(window=window_size).mean()
        satiety_norm_std = agent_data['satiety_norm'].rolling(window=window_size).std()
        
        # Count social touch events in each window
        social_touch_count = agent_data['social_touch'].rolling(window=window_size).apply(lambda x: (x > 0).sum())
        
        stats.append({
            'agent': agent,
            'arousal_mean': arousal_mean,
            'arousal_std': arousal_std,
            'satiety_mean': satiety_mean,
            'satiety_std': satiety_std,
            'arousal_norm_mean': arousal_norm_mean,
            'arousal_norm_std': arousal_norm_std,
            'satiety_norm_mean': satiety_norm_mean,
            'satiety_norm_std': satiety_norm_std,
            'social_touch_count': social_touch_count,
            't': agent_data['t']
        })
    return stats

def plot_metrics(df, stats, output_file_raw, output_file_norm):
    # Create figures for raw and normalized data
    fig_raw = make_subplots(rows=3, cols=1, subplot_titles=('Arousal Over Time', 'Satiety Over Time', 'Social Touch Events Over Time'))
    fig_norm = make_subplots(rows=3, cols=1, subplot_titles=('Normalized Arousal Over Time', 'Normalized Satiety Over Time', 'Social Touch Events Over Time'))

    # Colors for each agent
    colors = {'pursuer_0': 'blue', 'pursuer_1': 'red'}

    # Plot raw and normalized data
    for agent in df['agent'].unique():
        agent_data = df[df['agent'] == agent]
        
        # Raw data
        fig_raw.add_trace(go.Scatter(x=agent_data['t'], y=agent_data['arousal'], mode='lines', name=f'{agent} Arousal', line=dict(color=colors[agent])), row=1, col=1)
        fig_raw.add_trace(go.Scatter(x=agent_data['t'], y=agent_data['satiety'], mode='lines', name=f'{agent} Satiety', line=dict(color=colors[agent])), row=2, col=1)
        
        # Normalized data
        fig_norm.add_trace(go.Scatter(x=agent_data['t'], y=agent_data['arousal_norm'], mode='lines', name=f'{agent} Arousal', line=dict(color=colors[agent])), row=1, col=1)
        fig_norm.add_trace(go.Scatter(x=agent_data['t'], y=agent_data['satiety_norm'], mode='lines', name=f'{agent} Satiety', line=dict(color=colors[agent])), row=2, col=1)

    # Plot social touch event counts (same for both raw and normalized)
    for stat in stats:
        fig_raw.add_trace(go.Scatter(x=stat['t'], y=stat['social_touch_count'], mode='lines', name=f"{stat['agent']} Social Touch Events", line=dict(color=colors[stat['agent']])), row=3, col=1)
        fig_norm.add_trace(go.Scatter(x=stat['t'], y=stat['social_touch_count'], mode='lines', name=f"{stat['agent']} Social Touch Events", line=dict(color=colors[stat['agent']])), row=3, col=1)

    # Add rolling mean to raw and normalized plots
    for stat in stats:
        fig_raw.add_trace(go.Scatter(x=stat['t'], y=stat['arousal_mean'], mode='lines', name=f"{stat['agent']} Arousal Mean", line=dict(dash='dash', color=colors[stat['agent']])), row=1, col=1)
        fig_raw.add_trace(go.Scatter(x=stat['t'], y=stat['satiety_mean'], mode='lines', name=f"{stat['agent']} Satiety Mean", line=dict(dash='dash', color=colors[stat['agent']])), row=2, col=1)
        
        fig_norm.add_trace(go.Scatter(x=stat['t'], y=stat['arousal_norm_mean'], mode='lines', name=f"{stat['agent']} Arousal Mean", line=dict(dash='dash', color=colors[stat['agent']])), row=1, col=1)
        fig_norm.add_trace(go.Scatter(x=stat['t'], y=stat['satiety_norm_mean'], mode='lines', name=f"{stat['agent']} Satiety Mean", line=dict(dash='dash', color=colors[stat['agent']])), row=2, col=1)

    # Update layout for both figures
    for fig, title, output_file in [(fig_raw, "Raw Agent Metrics Over Time", output_file_raw), (fig_norm, "Normalized Agent Metrics Over Time", output_file_norm)]:
        fig.update_layout(height=900, width=1000, title_text=title)
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="Arousal" if "Raw" in title else "Normalized Arousal", row=1, col=1)
        fig.update_yaxes(title_text="Satiety" if "Raw" in title else "Normalized Satiety", row=2, col=1)
        fig.update_yaxes(title_text="Social Touch Event Count", row=3, col=1)

        # Save the figure as an interactive HTML file
        fig.write_html(output_file)
        print(f"Interactive plot saved as '{output_file}'")
    
    # Open the raw data plot in the default web browser
    webbrowser.open('file://' + os.path.realpath(output_file_raw))

def main(log_dir):
    input_file = os.path.join(log_dir, 'monitor.csv')
    transformed_file = os.path.join(log_dir, 'transformed_monitor.csv')
    plot_file_raw = os.path.join(log_dir, 'agent_metrics_plot_raw.html')
    plot_file_norm = os.path.join(log_dir, 'agent_metrics_plot_normalized.html')

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # Always transform the CSV to ensure we have the latest version with normalized columns
    df = transform_csv(input_file, transformed_file)

    # Calculate the window size for rolling statistics
    total_time_points = len(df['t'].unique())
    window_size = max(1, total_time_points // 100)  # Ensure at least 1

    # Calculate rolling statistics
    stats = calculate_rolling_stats(df, window_size)

    # Plot the metrics
    plot_metrics(df, stats, plot_file_raw, plot_file_norm)

    # Print simple metrics
    print("\nSimple Metrics:")
    for agent in df['agent'].unique():
        agent_data = df[df['agent'] == agent]
        print(f"\n{agent.capitalize()}:")
        print(f"Arousal - Mean: {agent_data['arousal'].mean():.4f}, Std Dev: {agent_data['arousal'].std():.4f}")
        print(f"Satiety - Mean: {agent_data['satiety'].mean():.4f}, Std Dev: {agent_data['satiety'].std():.4f}")
        print(f"Normalized Arousal - Mean: {agent_data['arousal_norm'].mean():.4f}, Std Dev: {agent_data['arousal_norm'].std():.4f}")
        print(f"Normalized Satiety - Mean: {agent_data['satiety_norm'].mean():.4f}, Std Dev: {agent_data['satiety_norm'].std():.4f}")
        print(f"Social Touch Events - Total: {(agent_data['social_touch'] > 0).sum()}")

    # Optionally, read and print the metadata
    with open(input_file, 'r') as f:
        first_line = f.readline().strip()
        if first_line.startswith('#'):
            metadata = json.loads(first_line[1:])
            print("\nMetadata:")
            print(metadata)

if __name__ == "__main__":
    # You can change this to any log directory you want to process
    log_dir = "./logs/pettingzoo.sisl.waterworld_model1_20240816-123210"
    main(log_dir)