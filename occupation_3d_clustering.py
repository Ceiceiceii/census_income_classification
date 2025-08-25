import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import silhouette_score
import seaborn as sns
import os

plt.style.use('seaborn-v0_8')
RANDOM_STATE = 42
SCALER_MODE = "quantile"

def make_numeric_scaler(mode: str):
    mode = mode.lower()
    if mode == "standard":
        return StandardScaler()
    if mode == "minmax":
        return MinMaxScaler()
    if mode == "robust":
        return RobustScaler(quantile_range=(25.0, 75.0))
    if mode == "quantile":
        return QuantileTransformer(output_distribution="normal", subsample=200000, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown SCALER_MODE: {mode}")

def load_and_prepare_data():
    df = pd.read_csv('census_data.csv')
    print(f"Loaded {df.shape[0]:,} samples with {df.shape[1]} columns")
    
    # Create target variable
    df['target'] = (~df['label'].astype(str).str.contains(r"-\s*50000", na=False)).astype(int)
    target_dist = df['target'].value_counts()
    print("Original distribution:")
    print(f" - ≤$50K: {target_dist.get(0,0):,} ({target_dist.get(0,0)/len(df)*100:.1f}%)")
    print(f" - >$50K: {target_dist.get(1,0):,} ({target_dist.get(1,0)/len(df)*100:.1f}%)")
    
    # Replace '?' -> NaN
    df = df.replace('?', np.nan)
    obj_cols = df.select_dtypes(include='object').columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()
    
    # Use same important features as classifier
    important_features = [
        'age', 'education', 'marital stat', 'sex', 'race',
        'weeks worked in year', 'capital gains', 'capital losses',
        'class of worker', 'major occupation code', 'num persons worked for employer'
    ]
    
    available_features = [f for f in important_features if f in df.columns]
    print(f" Selected {len(available_features)} key features (same as classifier)")
    
    df_sel = df[available_features + ['target']].copy()
    
    return df_sel

def calculate_occupation_features(df):
    # group by occupation and calculate key statistics
    occ_stats = df.groupby('major occupation code').agg({
        'age': ['mean', 'std', 'count'],
        'weeks worked in year': ['mean', 'std'],
        'capital gains': ['mean', 'median'],
        'capital losses': ['mean', 'median'], 
        'target': ['mean', 'count'],
        'num persons worked for employer': ['mean', 'std']
    }).round(2)
    
    # flatten column names
    occ_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in occ_stats.columns.values]
    
    # Filter occupations with meaningful sample size (>=100)
    meaningful_occs = occ_stats[occ_stats['target_count'] >= 100].copy()
    
    print(f"Analyzing {len(meaningful_occs)} occupation codes with ≥100 samples")
    
    # Create 3D clustering features
    clustering_features = pd.DataFrame({
        'avg_age': meaningful_occs['age_mean'],
        'avg_weeks_worked': meaningful_occs['weeks worked in year_mean'],
        'high_income_rate': meaningful_occs['target_mean'],
        'avg_capital_gains': meaningful_occs['capital gains_mean'],
        'sample_size': meaningful_occs['target_count']
    }, index=meaningful_occs.index)
    
    print("\nOccupation Features Summary:")
    print("Occupation                                Avg_Age  Weeks  Income_Rate  Samples")
    print("-" * 80)
    for occ, row in clustering_features.sort_values('high_income_rate', ascending=False).iterrows():
        print(f"{occ:<40} {row['avg_age']:7.1f} {row['avg_weeks_worked']:6.1f} {row['high_income_rate']:10.3f} {row['sample_size']:8.0f}")
    
    return clustering_features

def find_optimal_clusters(features_scaled, max_clusters=6):
    print("\nFINDING OPTIMAL NUMBER OF CLUSTERS:")
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_clusters + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette_avg = silhouette_score(features_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_avg:.3f}")
    
    # best K based on silhouette score
    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal clusters: K={best_k} (highest silhouette score: {max(silhouette_scores):.3f})")
    
    return best_k, inertias, silhouette_scores

def perform_3d_clustering(clustering_features):
    print("\nPERFORMING 3D CLUSTERING:")
    
    feature_cols = ['avg_age', 'avg_weeks_worked', 'high_income_rate']
    cluster_data = clustering_features[feature_cols].copy()
    
    scaler = make_numeric_scaler(SCALER_MODE)
    features_scaled = scaler.fit_transform(cluster_data)
    
    # optimal number of clusters
    optimal_k, inertias, silhouette_scores = find_optimal_clusters(features_scaled)
    
    # final clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Add cluster labels to data
    clustering_features['cluster'] = cluster_labels
    cluster_data['cluster'] = cluster_labels
    
    # Analyze clusters
    print(f"\nCLUSTER ANALYSIS (K={optimal_k}):")
    for cluster_id in range(optimal_k):
        cluster_occs = clustering_features[clustering_features['cluster'] == cluster_id]
        
        print(f"\nCluster {cluster_id}: {len(cluster_occs)} occupations")
        print(f" Avg Age: {cluster_occs['avg_age'].mean():.1f}")
        print(f" Avg Weeks: {cluster_occs['avg_weeks_worked'].mean():.1f}")
        print(f" Income Rate: {cluster_occs['high_income_rate'].mean():.3f}")
        print(f" Occupations: {list(cluster_occs.index)}")
    
    return cluster_data, clustering_features, scaler, kmeans

def create_3d_visualization(cluster_data, clustering_features):
    print("\nCREATING 3D VISUALIZATION:")
    
    # Set up the figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Color palette for 15 major occupation codes
    unique_occupations = sorted(cluster_data.index.unique())
    n_occupations = len(unique_occupations)
    print(f" Found {n_occupations} major occupation codes")
    
    # Use a diverse color palette for 15 groups
    colors = plt.cm.tab20(np.linspace(0, 1, n_occupations))
    occupation_color_map = {occ: colors[i] for i, occ in enumerate(unique_occupations)}
    
    # Main 3D scatter plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    for occupation in unique_occupations:
        occ_points = cluster_data.loc[[occupation]]
        
        # Size points by sample size (log scale for better visualization)
        sizes = np.log10(clustering_features.loc[occupation, 'sample_size']) * 50
        
        ax1.scatter(
            occ_points['avg_age'],
            occ_points['avg_weeks_worked'],
            occ_points['high_income_rate'],
            c=[occupation_color_map[occupation]],
            s=sizes,
            alpha=0.7,
            label=occupation[:20] + '...' if len(occupation) > 20 else occupation,
            edgecolors='black',
            linewidth=0.5
        )
    
    ax1.set_xlabel('Average Age (years)', fontsize=12)
    ax1.set_ylabel('Average Weeks Worked', fontsize=12)
    ax1.set_zlabel('High Income Rate', fontsize=12)
    ax1.set_title('3D Visualization: 15 Major Occupation Codes\\n(Point size = Sample Size)', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    for idx, (occ, row) in enumerate(cluster_data.iterrows()):
        if clustering_features.loc[occ, 'sample_size'] > 5000:  # Label major occupations
            ax1.text(row['avg_age'], row['avg_weeks_worked'], row['high_income_rate'],
                    f'  {occ[:15]}...', fontsize=8)
    
    projections = [
        ('avg_age', 'avg_weeks_worked', 'Age vs Weeks Worked'),
        ('avg_age', 'high_income_rate', 'Age vs Income Rate'),
        ('avg_weeks_worked', 'high_income_rate', 'Weeks Worked vs Income Rate')
    ]
    
    for i, (x_col, y_col, title) in enumerate(projections):
        ax = fig.add_subplot(2, 2, i + 2)
        
        for occupation in unique_occupations:
            occ_points = cluster_data.loc[[occupation]]
            
            sizes = np.log10(clustering_features.loc[occupation, 'sample_size']) * 30
            
            ax.scatter(
                occ_points[x_col],
                occ_points[y_col],
                c=[occupation_color_map[occupation]],
                s=sizes,
                alpha=0.7,
                label=occupation[:15] + '...' if len(occupation) > 15 else occupation,
                edgecolors='black',
                linewidth=0.5
            )
            
            # Add labels for major occupations
            if clustering_features.loc[occupation, 'sample_size'] > 8000:
                ax.annotate(occupation[:12] + '...', 
                          (occ_points[x_col].iloc[0], occ_points[y_col].iloc[0]),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, alpha=0.8)
        
        ax.set_xlabel(x_col.replace('_', ' ').title())
        ax.set_ylabel(y_col.replace('_', ' ').title())
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        # Remove legend from 2D projections to avoid clutter with 15 groups
    
    plt.tight_layout()
    
    # Create segmentation_plot directory if it doesn't exist
    os.makedirs('segmentation_plot', exist_ok=True)
    
    plt.savefig('segmentation_plot/occupation_3d_clustering.png', dpi=300, bbox_inches='tight')
    print(" 3D visualization saved as 'segmentation_plot/occupation_3d_clustering.png'")
    plt.show()
    
    return fig

def analyze_occupation_group_differences(clustering_features):
    print("\n" + "="*80)
    print("OCCUPATION GROUP DIFFERENCE ANALYSIS")
    print("="*80)
    
    # Sort by high income rate for better analysis
    sorted_occupations = clustering_features.sort_values('high_income_rate', ascending=False)
    
    print("\n OCCUPATION GROUPS RANKED BY INCOME POTENTIAL:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Occupation':<35} {'Income Rate':<12} {'Avg Age':<8} {'Weeks':<6} {'Samples':<8}")
    print("-" * 80)
    
    for i, (occ, row) in enumerate(sorted_occupations.iterrows(), 1):
        print(f"{i:<4} {occ[:34]:<35} {row['high_income_rate']:8.3f}     {row['avg_age']:6.1f}  {row['avg_weeks_worked']:5.1f}  {row['sample_size']:7,.0f}")
    
    # Statistical analysis of groups
    print(f"\nKEY STATISTICS ACROSS 15 OCCUPATION GROUPS:")
    print("-" * 60)
    stats = clustering_features.describe().round(3)
    for col in ['high_income_rate', 'avg_age', 'avg_weeks_worked', 'sample_size']:
        print(f"\n{col.replace('_', ' ').title()}:")
        print(f"  Mean: {stats.loc['mean', col]:.3f}")
        print(f"  Std:  {stats.loc['std', col]:.3f}")
        print(f"  Min:  {stats.loc['min', col]:.3f}")
        print(f"  Max:  {stats.loc['max', col]:.3f}")
    
    # Identify distinct groups
    high_income_threshold = clustering_features['high_income_rate'].quantile(0.75)
    low_income_threshold = clustering_features['high_income_rate'].quantile(0.25)
    
    high_income_occs = clustering_features[clustering_features['high_income_rate'] >= high_income_threshold]
    mid_income_occs = clustering_features[
        (clustering_features['high_income_rate'] < high_income_threshold) & 
        (clustering_features['high_income_rate'] > low_income_threshold)
    ]
    low_income_occs = clustering_features[clustering_features['high_income_rate'] <= low_income_threshold]
    
    print(f"\nHIGH-INCOME OCCUPATIONS (Top 25%, ≥{high_income_threshold:.3f} income rate):")
    print("-" * 70)
    for occ, row in high_income_occs.sort_values('high_income_rate', ascending=False).iterrows():
        print(f" - {occ:<40} {row['high_income_rate']:.3f} ({row['sample_size']:,} samples)")
    
    print(f"\nMID-INCOME OCCUPATIONS (Middle 50%):")
    print("-" * 70)
    for occ, row in mid_income_occs.sort_values('high_income_rate', ascending=False).iterrows():
        print(f" • {occ:<40} {row['high_income_rate']:.3f} ({row['sample_size']:,} samples)")
    
    print(f"\nLOW-INCOME OCCUPATIONS (Bottom 25%, ≤{low_income_threshold:.3f} income rate):")
    print("-" * 70)
    for occ, row in low_income_occs.sort_values('high_income_rate', ascending=False).iterrows():
        print(f" - {occ:<40} {row['high_income_rate']:.3f} ({row['sample_size']:,} samples)")
    
    return {
        'high_income_group': high_income_occs,
        'mid_income_group': mid_income_occs, 
        'low_income_group': low_income_occs,
        'sorted_occupations': sorted_occupations
    }

def main():
    df = load_and_prepare_data()
    
    # occupation features
    clustering_features = calculate_occupation_features(df)
    
    # 3D clustering
    cluster_data, enhanced_features, scaler, kmeans = perform_3d_clustering(clustering_features)
    
    # visualizations
    fig = create_3d_visualization(cluster_data, enhanced_features)
    
    #how the 15 occupation groups differ from each other
    group_analysis = analyze_occupation_group_differences(clustering_features)
    
    #results
    enhanced_features.to_csv('occupation_clustering_results.csv')
    group_analysis['sorted_occupations'].to_csv('occupation_marketing_analysis.csv')   
    return {
        'occupation_features': clustering_features,
        'cluster_data': cluster_data,
        'group_analysis': group_analysis,
        'scaler': scaler,
        'kmeans': kmeans
    }

if __name__ == "__main__":
    results = main()
