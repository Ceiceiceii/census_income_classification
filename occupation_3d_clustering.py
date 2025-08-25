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

plt.style.use('seaborn-v0_8')
RANDOM_STATE = 42
SCALER_MODE = "quantile"  # Same as classifier

def make_numeric_scaler(mode: str):
    """Create the same scaler as used in the classifier."""
    mode = mode.lower()
    if mode == "standard":
        return StandardScaler()
    if mode == "minmax":
        return MinMaxScaler()
    if mode == "robust":
        return RobustScaler(quantile_range=(25.0, 75.0))
    if mode == "quantile":
        # map to ~Normal — often great for skewed heavy-tail numerics with LR
        return QuantileTransformer(output_distribution="normal", subsample=200000, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown SCALER_MODE: {mode}")

def load_and_prepare_data():
    """Load census data and prepare for occupation analysis using same approach as classifier."""
    print("="*70)
    print("3D CLUSTERING ANALYSIS: MAJOR OCCUPATION CODES")
    print("(Using same data processing as classifier)")
    print("="*70)
    
    df = pd.read_csv('census_data.csv')
    print(f"✓ Loaded {df.shape[0]:,} samples with {df.shape[1]} columns")
    
    # Create target variable (same as classifier)
    df['target'] = (~df['label'].astype(str).str.contains(r"-\s*50000", na=False)).astype(int)
    target_dist = df['target'].value_counts()
    print("✓ Original distribution:")
    print(f"  - ≤$50K: {target_dist.get(0,0):,} ({target_dist.get(0,0)/len(df)*100:.1f}%)")
    print(f"  - >$50K: {target_dist.get(1,0):,} ({target_dist.get(1,0)/len(df)*100:.1f}%)")
    
    # Replace '?' -> NaN, trim strings (same as classifier)
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
    print(f"✓ Selected {len(available_features)} key features (same as classifier)")
    
    df_sel = df[available_features + ['target']].copy()
    
    # Apply same outlier removal as classifier
    print("\n🧹 OUTLIER REMOVAL (same as classifier):")
    initial_count = len(df_sel)
    
    # Remove invalid ages (age == 0)
    age_outliers = (df_sel['age'] == 0)
    if age_outliers.any():
        removed_age = age_outliers.sum()
        print(f"  - Removing {removed_age:,} samples with invalid age (age=0)")
        df_sel = df_sel[~age_outliers]
    
    # Remove extreme capital gains/losses outliers (above 99th percentile)
    for col in ['capital gains', 'capital losses']:
        if col in df_sel.columns:
            q99 = df_sel[col].quantile(0.99)
            extreme_mask = df_sel[col] > q99
            if extreme_mask.any():
                removed_extreme = extreme_mask.sum()
                print(f"  - Removing {removed_extreme:,} samples with extreme {col} (>{q99:,.0f})")
                df_sel = df_sel[~extreme_mask]
    
    final_count = len(df_sel)
    total_removed = initial_count - final_count
    print(f"✓ Outlier removal complete: {total_removed:,} samples removed ({total_removed/initial_count*100:.2f}%)")
    print(f"✓ Remaining samples: {final_count:,}")
    
    return df_sel

def calculate_occupation_features(df):
    """Calculate 3D features for each occupation code."""
    print("\n📊 CALCULATING OCCUPATION FEATURES:")
    
    # Group by occupation and calculate key statistics
    occ_stats = df.groupby('major occupation code').agg({
        'age': ['mean', 'std', 'count'],
        'weeks worked in year': ['mean', 'std'],
        'capital gains': ['mean', 'median'],
        'capital losses': ['mean', 'median'], 
        'target': ['mean', 'count'],
        'num persons worked for employer': ['mean', 'std']
    }).round(2)
    
    # Flatten column names
    occ_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in occ_stats.columns.values]
    
    # Filter occupations with meaningful sample size (>=100)
    meaningful_occs = occ_stats[occ_stats['target_count'] >= 100].copy()
    
    print(f"✓ Analyzing {len(meaningful_occs)} occupation codes with ≥100 samples")
    
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
    """Find optimal number of clusters using elbow method and silhouette score."""
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
        
        print(f"  K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_avg:.3f}")
    
    # Find best K based on silhouette score
    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"✓ Optimal clusters: K={best_k} (highest silhouette score: {max(silhouette_scores):.3f})")
    
    return best_k, inertias, silhouette_scores

def perform_3d_clustering(clustering_features):
    """Perform 3D clustering on occupation codes using same scaler as classifier."""
    print("\n🎯 PERFORMING 3D CLUSTERING:")
    
    # Select the 3 main dimensions for clustering
    feature_cols = ['avg_age', 'avg_weeks_worked', 'high_income_rate']
    cluster_data = clustering_features[feature_cols].copy()
    
    print("✓ 3D Clustering Dimensions:")
    print("  1. Average Age (professional maturity)")
    print("  2. Average Weeks Worked (employment stability)")
    print("  3. High Income Rate (earning potential)")
    
    # Use same scaler as classifier
    scaler = make_numeric_scaler(SCALER_MODE)
    print(f"✓ Using {SCALER_MODE} scaler (same as classifier)")
    features_scaled = scaler.fit_transform(cluster_data)
    
    # Find optimal number of clusters
    optimal_k, inertias, silhouette_scores = find_optimal_clusters(features_scaled)
    
    # Perform final clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Add cluster labels to data
    clustering_features['cluster'] = cluster_labels
    cluster_data['cluster'] = cluster_labels
    
    # Analyze clusters
    print(f"\n📊 CLUSTER ANALYSIS (K={optimal_k}):")
    for cluster_id in range(optimal_k):
        cluster_occs = clustering_features[clustering_features['cluster'] == cluster_id]
        
        print(f"\n🏷️  Cluster {cluster_id}: {len(cluster_occs)} occupations")
        print(f"   📈 Avg Age: {cluster_occs['avg_age'].mean():.1f}")
        print(f"   📈 Avg Weeks: {cluster_occs['avg_weeks_worked'].mean():.1f}")
        print(f"   📈 Income Rate: {cluster_occs['high_income_rate'].mean():.3f}")
        print(f"   📋 Occupations: {list(cluster_occs.index)}")
    
    return cluster_data, clustering_features, scaler, kmeans

def create_3d_visualization(cluster_data, clustering_features):
    """Create comprehensive 3D visualization of occupation clusters."""
    print("\n🎨 CREATING 3D VISUALIZATION:")
    
    # Set up the figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Color palette for 15 major occupation codes
    unique_occupations = sorted(cluster_data.index.unique())
    n_occupations = len(unique_occupations)
    print(f"✓ Found {n_occupations} major occupation codes")
    
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
    
    # Add occupation labels for key points
    for idx, (occ, row) in enumerate(cluster_data.iterrows()):
        if clustering_features.loc[occ, 'sample_size'] > 5000:  # Label major occupations
            ax1.text(row['avg_age'], row['avg_weeks_worked'], row['high_income_rate'],
                    f'  {occ[:15]}...', fontsize=8)
    
    # 2D projections
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
    plt.savefig('occupation_3d_clustering.png', dpi=300, bbox_inches='tight')
    print("✓ 3D visualization saved as 'occupation_3d_clustering.png'")
    plt.show()
    
    return fig

def analyze_occupation_group_differences(clustering_features):
    """Analyze how the 15 major occupation groups differ from each other."""
    print("\n" + "="*80)
    print("📊 OCCUPATION GROUP DIFFERENCE ANALYSIS")
    print("="*80)
    
    # Sort by high income rate for better analysis
    sorted_occupations = clustering_features.sort_values('high_income_rate', ascending=False)
    
    print("\n🎯 OCCUPATION GROUPS RANKED BY INCOME POTENTIAL:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Occupation':<35} {'Income Rate':<12} {'Avg Age':<8} {'Weeks':<6} {'Samples':<8}")
    print("-" * 80)
    
    for i, (occ, row) in enumerate(sorted_occupations.iterrows(), 1):
        print(f"{i:<4} {occ[:34]:<35} {row['high_income_rate']:8.3f}     {row['avg_age']:6.1f}  {row['avg_weeks_worked']:5.1f}  {row['sample_size']:7,.0f}")
    
    # Statistical analysis of groups
    print(f"\n📈 KEY STATISTICS ACROSS 15 OCCUPATION GROUPS:")
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
    
    print(f"\n🏆 HIGH-INCOME OCCUPATIONS (Top 25%, ≥{high_income_threshold:.3f} income rate):")
    print("-" * 70)
    for occ, row in high_income_occs.sort_values('high_income_rate', ascending=False).iterrows():
        print(f"  • {occ:<40} {row['high_income_rate']:.3f} ({row['sample_size']:,} samples)")
    
    print(f"\n🏢 MID-INCOME OCCUPATIONS (Middle 50%):")
    print("-" * 70)
    for occ, row in mid_income_occs.sort_values('high_income_rate', ascending=False).iterrows():
        print(f"  • {occ:<40} {row['high_income_rate']:.3f} ({row['sample_size']:,} samples)")
    
    print(f"\n📉 LOW-INCOME OCCUPATIONS (Bottom 25%, ≤{low_income_threshold:.3f} income rate):")
    print("-" * 70)
    for occ, row in low_income_occs.sort_values('high_income_rate', ascending=False).iterrows():
        print(f"  • {occ:<40} {row['high_income_rate']:.3f} ({row['sample_size']:,} samples)")
    
    return {
        'high_income_group': high_income_occs,
        'mid_income_group': mid_income_occs, 
        'low_income_group': low_income_occs,
        'sorted_occupations': sorted_occupations
    }

def generate_retail_marketing_insights(clustering_features, group_analysis):
    """Generate retail marketing insights and recommendations for each occupation group."""
    print("\n" + "="*80)
    print("🛍️  RETAIL MARKETING STRATEGY: OCCUPATION-BASED CUSTOMER SEGMENTATION")
    print("="*80)
    
    print("\n💡 EXECUTIVE SUMMARY:")
    print("-" * 50)
    total_samples = clustering_features['sample_size'].sum()
    avg_income_rate = clustering_features['high_income_rate'].mean()
    
    print(f"✓ Analyzed {len(clustering_features)} distinct occupation groups")
    print(f"✓ Total market size: {total_samples:,} potential customers") 
    print(f"✓ Average high-income rate: {avg_income_rate:.1%}")
    print(f"✓ Income potential ranges from {clustering_features['high_income_rate'].min():.1%} to {clustering_features['high_income_rate'].max():.1%}")
    
    # Marketing strategies for each tier
    high_income = group_analysis['high_income_group']
    mid_income = group_analysis['mid_income_group']
    low_income = group_analysis['low_income_group']
    
    print(f"\nPREMIUM CUSTOMER SEGMENT (HIGH-INCOME OCCUPATIONS)")
    print("="*70)
    print(f"Market Size: {high_income['sample_size'].sum():,} customers ({high_income['sample_size'].sum()/total_samples:.1%} of market)")
    print(f"Income Characteristics: {high_income['high_income_rate'].mean():.1%} average high-income rate")
    print(f"Age Profile: {high_income['avg_age'].mean():.1f} years average")
    print(f"Work Pattern: {high_income['avg_weeks_worked'].mean():.1f} weeks/year average")
    
    print(f"\nMARKETING STRATEGY:")
    print("   • PREMIUM PRODUCTS: Luxury items, high-end electronics, premium brands")
    print("   • PRICING: Premium pricing strategy, focus on quality over price")
    print("   • CHANNELS: Upscale retail locations, online premium platforms")
    print("   • MESSAGING: Quality, status, convenience, time-saving benefits")
    print("   • TIMING: Focus on evenings and weekends (working professionals)")
    
    print(f"\n📋 TARGET OCCUPATIONS:")
    for occ, row in high_income.sort_values('sample_size', ascending=False).iterrows():
        market_share = row['sample_size'] / total_samples * 100
        print(f"   • {occ:<35} ({row['sample_size']:,} customers, {market_share:.1f}% market share)")
    
    print(f"\n🎯 MAINSTREAM CUSTOMER SEGMENT (MID-INCOME OCCUPATIONS)")
    print("="*70)
    print(f"📊 Market Size: {mid_income['sample_size'].sum():,} customers ({mid_income['sample_size'].sum()/total_samples:.1%} of market)")
    print(f"💰 Income Characteristics: {mid_income['high_income_rate'].mean():.1%} average high-income rate")
    print(f"👥 Age Profile: {mid_income['avg_age'].mean():.1f} years average")
    print(f"💼 Work Pattern: {mid_income['avg_weeks_worked'].mean():.1f} weeks/year average")
    
    print(f"\n🛍️  MARKETING STRATEGY:")
    print("   • VALUE PRODUCTS: Mid-range quality, good value for money")
    print("   • PRICING: Competitive pricing, sales and promotions")  
    print("   • CHANNELS: Department stores, online marketplaces, suburban malls")
    print("   • MESSAGING: Value, reliability, family-focused, practical benefits")
    print("   • PROMOTIONS: Seasonal sales, loyalty programs, bundled offers")
    
    print(f"\n📋 TARGET OCCUPATIONS:")
    for occ, row in mid_income.sort_values('sample_size', ascending=False).iterrows():
        market_share = row['sample_size'] / total_samples * 100
        print(f"   • {occ:<35} ({row['sample_size']:,} customers, {market_share:.1f}% market share)")
    
    print(f"\n🎯 VALUE CUSTOMER SEGMENT (BUDGET-CONSCIOUS OCCUPATIONS)")
    print("="*70)
    print(f"📊 Market Size: {low_income['sample_size'].sum():,} customers ({low_income['sample_size'].sum()/total_samples:.1%} of market)")
    print(f"💰 Income Characteristics: {low_income['high_income_rate'].mean():.1%} average high-income rate")
    print(f"👥 Age Profile: {low_income['avg_age'].mean():.1f} years average")
    print(f"💼 Work Pattern: {low_income['avg_weeks_worked'].mean():.1f} weeks/year average")
    
    print(f"\n🛍️  MARKETING STRATEGY:")
    print("   • BUDGET PRODUCTS: Affordable options, generic brands, essentials")
    print("   • PRICING: Low prices, deep discounts, clearance items")
    print("   • CHANNELS: Discount stores, dollar stores, online deals sites")
    print("   • MESSAGING: Savings, affordability, necessity, practical value")
    print("   • PROMOTIONS: Deep discounts, buy-one-get-one, cashback offers")
    
    print(f"\n📋 TARGET OCCUPATIONS:")
    for occ, row in low_income.sort_values('sample_size', ascending=False).iterrows():
        market_share = row['sample_size'] / total_samples * 100
        print(f"   • {occ:<35} ({row['sample_size']:,} customers, {market_share:.1f}% market share)")
    
    # Overall recommendations
    print(f"\n🚀 KEY RECOMMENDATIONS FOR RETAIL CLIENT:")
    print("="*60)
    print("1. 📊 PORTFOLIO STRATEGY:")
    print(f"   • Allocate {high_income['sample_size'].sum()/total_samples:.0%} of premium inventory to high-income segments")
    print(f"   • Focus {mid_income['sample_size'].sum()/total_samples:.0%} of mainstream products on mid-income segments")
    print(f"   • Develop {low_income['sample_size'].sum()/total_samples:.0%} of budget offerings for price-sensitive segments")
    
    print("\n2. 🎯 TARGETED CAMPAIGNS:")
    largest_segment = clustering_features.loc[clustering_features['sample_size'].idxmax()]
    highest_income = clustering_features.loc[clustering_features['high_income_rate'].idxmax()]
    print(f"   • Prioritize '{largest_segment.name}' (largest segment: {largest_segment['sample_size']:,} customers)")
    print(f"   • Premium focus on '{highest_income.name}' (highest income rate: {highest_income['high_income_rate']:.1%})")
    
    print("\n3. 📱 DIGITAL MARKETING:")
    stable_workers = clustering_features[clustering_features['avg_weeks_worked'] > 45]
    print(f"   • Target working professionals ({len(stable_workers)} occupation groups) during off-hours")
    print("   • Use LinkedIn advertising for professional services and executive roles")
    print("   • Implement geo-targeting based on occupation density in different areas")
    
    print("\n4. 🏪 STORE STRATEGY:")
    print("   • Premium locations near business districts for high-income occupations")
    print("   • Suburban locations for family-oriented mid-income segments") 
    print("   • Urban and accessible locations for budget-conscious segments")
    
    return {
        'high_income_strategy': high_income,
        'mid_income_strategy': mid_income,
        'low_income_strategy': low_income,
        'total_market_size': total_samples
    }

def main():
    """Run complete 3D clustering analysis for occupation codes with retail marketing insights."""
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Calculate occupation features
    clustering_features = calculate_occupation_features(df)
    
    # Perform 3D clustering (but focus on occupation groups, not clusters)
    cluster_data, enhanced_features, scaler, kmeans = perform_3d_clustering(clustering_features)
    
    # Create visualizations (now showing 15 occupation groups with different colors)
    fig = create_3d_visualization(cluster_data, enhanced_features)
    
    # Analyze how the 15 occupation groups differ from each other
    group_analysis = analyze_occupation_group_differences(clustering_features)
    
    # Generate retail marketing insights and recommendations
    marketing_insights = generate_retail_marketing_insights(clustering_features, group_analysis)
    
    # Save results
    enhanced_features.to_csv('occupation_clustering_results.csv')
    group_analysis['sorted_occupations'].to_csv('occupation_marketing_analysis.csv')
    print("\n💾 Results saved to:")
    print("   • occupation_clustering_results.csv (technical results)")
    print("   • occupation_marketing_analysis.csv (marketing analysis)")
    
    print("\n" + "="*70)
    print("OCCUPATION-BASED RETAIL MARKETING ANALYSIS COMPLETE")
    print("="*70)
    print("✓ 15 Major occupation groups analyzed with distinct colors")
    print("✓ 3D visualization shows: Age × Weeks Worked × Income Rate")
    print("✓ Group differences identified across income, age, and work patterns")
    print("✓ Retail marketing strategies developed for each customer segment")
    print("✓ Actionable recommendations provided for product portfolio & targeting")
    
    return {
        'occupation_features': clustering_features,
        'cluster_data': cluster_data,
        'group_analysis': group_analysis,
        'marketing_insights': marketing_insights,
        'scaler': scaler,
        'kmeans': kmeans
    }

if __name__ == "__main__":
    results = main()
