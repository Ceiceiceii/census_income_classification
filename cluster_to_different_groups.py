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
SCALER_MODE = "quantile"

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
        return QuantileTransformer(output_distribution="normal", subsample=200000, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown SCALER_MODE: {mode}")

def load_and_prepare_weighted_data():
    """Load census data and prepare for weighted analysis."""
    print("="*80)
    print("ENHANCED WEIGHTED CLUSTERING ANALYSIS FOR RETAIL MARKETING")
    print("Incorporating Population Weights & Geographic Distribution")
    print("="*80)
    
    df = pd.read_csv('census_data.csv')
    print(f"✓ Loaded {df.shape[0]:,} samples with {df.shape[1]} columns")
    
    # Check if weight column exists
    if 'weight' not in df.columns:
        # Look for similar column names
        weight_cols = [col for col in df.columns if 'weight' in col.lower() or 'wt' in col.lower()]
        if weight_cols:
            df['weight'] = df[weight_cols[0]]
            print(f"✓ Using '{weight_cols[0]}' as weight column")
        else:
            print("⚠️ No weight column found - using uniform weights")
            df['weight'] = 1.0
    
    # Create target variable
    df['target'] = (~df['label'].astype(str).str.contains(r"-\s*50000", na=False)).astype(int)
    
    # Calculate weighted distribution
    total_population = df['weight'].sum()
    weighted_dist = df.groupby('target')['weight'].sum()
    print(f"\n✓ WEIGHTED Population Distribution (Total: {total_population:,.0f}):")
    print(f"  - ≤$50K: {weighted_dist.get(0,0):,.0f} ({weighted_dist.get(0,0)/total_population*100:.1f}%)")
    print(f"  - >$50K: {weighted_dist.get(1,0):,.0f} ({weighted_dist.get(1,0)/total_population*100:.1f}%)")
    
    # Also show unweighted for comparison
    unweighted_dist = df['target'].value_counts()
    print(f"\n📊 Unweighted Sample Distribution:")
    print(f"  - ≤$50K: {unweighted_dist.get(0,0):,} ({unweighted_dist.get(0,0)/len(df)*100:.1f}%)")
    print(f"  - >$50K: {unweighted_dist.get(1,0):,} ({unweighted_dist.get(1,0)/len(df)*100:.1f}%)")
    
    # Replace '?' -> NaN, trim strings
    df = df.replace('?', np.nan)
    obj_cols = df.select_dtypes(include='object').columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()
    
    # Enhanced feature selection including geographic and demographic factors
    enhanced_features = [
        'age', 'education', 'marital stat', 'sex', 'race',
        'weeks worked in year', 'capital gains', 'capital losses',
        'class of worker', 'major occupation code', 'num persons worked for employer',
        'weight'  # Include weight for population representation
    ]
    
    # Add geographic features if available
    geographic_features = [col for col in df.columns if any(geo in col.lower() 
                          for geo in ['state', 'region', 'msa', 'metro', 'urban', 'rural'])]
    
    if geographic_features:
        enhanced_features.extend(geographic_features)
        print(f"✓ Found geographic features: {geographic_features}")
    
    # Add additional demographic features if available
    additional_demo_features = [col for col in df.columns if any(demo in col.lower() 
                               for demo in ['household', 'family', 'children', 'veteran', 'disability'])]
    
    if additional_demo_features:
        enhanced_features.extend(additional_demo_features)
        print(f"✓ Found additional demographic features: {additional_demo_features}")
    
    available_features = [f for f in enhanced_features if f in df.columns]
    print(f"✓ Selected {len(available_features)} features for analysis")
    
    df_sel = df[available_features + ['target']].copy()
    
    # Enhanced outlier removal with weight consideration
    print(f"\n🧹 WEIGHTED OUTLIER REMOVAL:")
    initial_count = len(df_sel)
    initial_population = df_sel['weight'].sum()
    
    # Remove samples with extreme weights (potential data errors)
    weight_q99 = df_sel['weight'].quantile(0.99)
    weight_outliers = df_sel['weight'] > weight_q99
    if weight_outliers.any():
        removed_weight = weight_outliers.sum()
        removed_pop = df_sel[weight_outliers]['weight'].sum()
        print(f"  - Removing {removed_weight:,} samples with extreme weights (>{weight_q99:.0f})")
        print(f"    Population impact: {removed_pop:,.0f} ({removed_pop/initial_population*100:.2f}%)")
        df_sel = df_sel[~weight_outliers]
    
    # Remove invalid ages
    age_outliers = (df_sel['age'] == 0)
    if age_outliers.any():
        removed_age = age_outliers.sum()
        removed_pop_age = df_sel[age_outliers]['weight'].sum()
        print(f"  - Removing {removed_age:,} samples with invalid age")
        print(f"    Population impact: {removed_pop_age:,.0f} ({removed_pop_age/initial_population*100:.2f}%)")
        df_sel = df_sel[~age_outliers]
    
    # Remove extreme capital gains/losses outliers
    for col in ['capital gains', 'capital losses']:
        if col in df_sel.columns:
            q99 = df_sel[col].quantile(0.99)
            extreme_mask = df_sel[col] > q99
            if extreme_mask.any():
                removed_extreme = extreme_mask.sum()
                removed_pop_extreme = df_sel[extreme_mask]['weight'].sum()
                print(f"  - Removing {removed_extreme:,} samples with extreme {col}")
                print(f"    Population impact: {removed_pop_extreme:,.0f} ({removed_pop_extreme/initial_population*100:.2f}%)")
                df_sel = df_sel[~extreme_mask]
    
    final_count = len(df_sel)
    final_population = df_sel['weight'].sum()
    print(f"✓ Final dataset: {final_count:,} samples representing {final_population:,.0f} population")
    print(f"✓ Retention rate: {final_count/initial_count*100:.1f}% (samples), {final_population/initial_population*100:.1f}% (population)")
    
    return df_sel

def calculate_weighted_occupation_features(df):
    """Calculate weighted features for each occupation code."""
    print("\n📊 CALCULATING WEIGHTED OCCUPATION FEATURES:")
    
    # Group by occupation and calculate weighted statistics
    occupation_stats = []
    
    for occ in df['major occupation code'].unique():
        if pd.isna(occ):
            continue
            
        occ_data = df[df['major occupation code'] == occ].copy()
        
        if len(occ_data) < 50:  # Skip occupations with too few samples
            continue
        
        # Calculate weighted statistics
        total_weight = occ_data['weight'].sum()
        
        stats = {
            'occupation': occ,
            'sample_size': len(occ_data),
            'population_represented': total_weight,
            
            # Weighted averages
            'weighted_avg_age': np.average(occ_data['age'], weights=occ_data['weight']),
            'weighted_avg_weeks': np.average(occ_data['weeks worked in year'], weights=occ_data['weight']),
            'weighted_avg_capital_gains': np.average(occ_data['capital gains'], weights=occ_data['weight']),
            'weighted_avg_capital_losses': np.average(occ_data['capital losses'], weights=occ_data['weight']),
            
            # Weighted income rate
            'weighted_high_income_rate': np.average(occ_data['target'], weights=occ_data['weight']),
            
            # Market potential metrics
            'high_income_population': np.sum(occ_data[occ_data['target'] == 1]['weight']),
            'low_income_population': np.sum(occ_data[occ_data['target'] == 0]['weight']),
            
            # Diversity metrics
            'age_std': np.sqrt(np.average((occ_data['age'] - np.average(occ_data['age'], weights=occ_data['weight']))**2, 
                                        weights=occ_data['weight'])),
        }
        
        # Add geographic diversity if available
        geographic_features = [col for col in df.columns if any(geo in col.lower() 
                              for geo in ['state', 'region', 'msa'])]
        
        if geographic_features:
            for geo_col in geographic_features:
                if geo_col in occ_data.columns:
                    # Calculate geographic spread
                    geo_counts = occ_data.groupby(geo_col)['weight'].sum()
                    geo_entropy = -np.sum((geo_counts / geo_counts.sum()) * np.log2(geo_counts / geo_counts.sum() + 1e-10))
                    stats[f'{geo_col}_diversity'] = geo_entropy
        
        occupation_stats.append(stats)
    
    occ_df = pd.DataFrame(occupation_stats).set_index('occupation')
    occ_df = occ_df.sort_values('population_represented', ascending=False)
    
    print(f"✓ Analyzed {len(occ_df)} occupation codes with ≥50 samples")
    print(f"✓ Total population represented: {occ_df['population_represented'].sum():,.0f}")
    
    # Display top occupations by population
    print(f"\n🏆 TOP 10 OCCUPATIONS BY POPULATION SIZE:")
    print("-" * 85)
    print(f"{'Occupation':<35} {'Population':<12} {'Income Rate':<12} {'Avg Age':<8} {'Samples'}")
    print("-" * 85)
    
    for occ, row in occ_df.head(10).iterrows():
        print(f"{occ[:34]:<35} {row['population_represented']:>10,.0f} "
              f"{row['weighted_high_income_rate']:>10.3f} "
              f"{row['weighted_avg_age']:>7.1f} {row['sample_size']:>7,.0f}")
    
    return occ_df

def perform_weighted_clustering_analysis(occ_df):
    """Perform clustering analysis using weighted features."""
    print(f"\n🎯 WEIGHTED CLUSTERING ANALYSIS:")
    
    # Prepare clustering features with population weights
    clustering_features = pd.DataFrame({
        'weighted_avg_age': occ_df['weighted_avg_age'],
        'weighted_avg_weeks': occ_df['weighted_avg_weeks'],
        'weighted_high_income_rate': occ_df['weighted_high_income_rate'],
        'log_population': np.log10(occ_df['population_represented']),
        'age_diversity': occ_df['age_std'],
        'market_potential_score': (occ_df['weighted_high_income_rate'] * 
                                 np.log10(occ_df['population_represented']))
    })
    
    print("✓ Clustering dimensions:")
    print("  1. Weighted Average Age (professional maturity)")
    print("  2. Weighted Average Weeks Worked (employment stability)")  
    print("  3. Weighted High Income Rate (earning potential)")
    print("  4. Log Population Size (market reach)")
    print("  5. Age Diversity (market heterogeneity)")
    print("  6. Market Potential Score (income rate × log population)")
    
    # Scale features
    scaler = make_numeric_scaler(SCALER_MODE)
    features_scaled = scaler.fit_transform(clustering_features)
    
    # Find optimal clusters using weighted silhouette score
    silhouette_scores = []
    k_range = range(3, 8)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Weight silhouette score by population
        sample_weights = occ_df['population_represented'].values
        sil_score = silhouette_score(features_scaled, cluster_labels)
        silhouette_scores.append(sil_score)
        
        print(f"  K={k}: Weighted Silhouette Score={sil_score:.3f}")
    
    # Select optimal K
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"✓ Optimal clusters: K={optimal_k}")
    
    # Final clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    occ_df['cluster'] = cluster_labels
    clustering_features['cluster'] = cluster_labels
    
    return clustering_features, occ_df, scaler, kmeans, optimal_k

def analyze_weighted_market_segments(occ_df, optimal_k):
    """Analyze market segments using population weights."""
    print(f"\n📈 WEIGHTED MARKET SEGMENT ANALYSIS:")
    print("="*80)
    
    total_population = occ_df['population_represented'].sum()
    segment_analysis = {}
    
    for cluster_id in range(optimal_k):
        cluster_data = occ_df[occ_df['cluster'] == cluster_id].copy()
        cluster_pop = cluster_data['population_represented'].sum()
        cluster_high_income_pop = cluster_data['high_income_population'].sum()
        
        segment_info = {
            'occupations': len(cluster_data),
            'population': cluster_pop,
            'market_share': cluster_pop / total_population,
            'high_income_population': cluster_high_income_pop,
            'avg_weighted_age': np.average(cluster_data['weighted_avg_age'], 
                                         weights=cluster_data['population_represented']),
            'avg_weighted_weeks': np.average(cluster_data['weighted_avg_weeks'],
                                           weights=cluster_data['population_represented']),
            'avg_income_rate': np.average(cluster_data['weighted_high_income_rate'],
                                        weights=cluster_data['population_represented']),
            'top_occupations': cluster_data.nlargest(5, 'population_represented')[['population_represented', 'weighted_high_income_rate']]
        }
        
        segment_analysis[cluster_id] = segment_info
        
        print(f"\n🏷️  MARKET SEGMENT {cluster_id}:")
        print(f"   📊 Population: {cluster_pop:,.0f} ({segment_info['market_share']:.1%} of total market)")
        print(f"   💰 High-Income Population: {cluster_high_income_pop:,.0f}")
        print(f"   👥 Demographics: {segment_info['avg_weighted_age']:.1f} years avg, {segment_info['avg_weighted_weeks']:.1f} weeks/year")
        print(f"   📈 Income Rate: {segment_info['avg_income_rate']:.1%}")
        print(f"   🎯 Occupations: {segment_info['occupations']} different occupation types")
        
        print(f"   🏆 Top 3 Occupations by Population:")
        for occ, row in segment_info['top_occupations'].head(3).iterrows():
            occ_share = row['population_represented'] / cluster_pop
            print(f"      • {occ[:30]:<30} {row['population_represented']:>8,.0f} ({occ_share:.1%}) - {row['weighted_high_income_rate']:.1%} income rate")
    
    return segment_analysis

def generate_enhanced_marketing_strategy(occ_df, segment_analysis, optimal_k):
    """Generate comprehensive marketing strategy using weighted analysis."""
    print(f"\n🛍️  ENHANCED RETAIL MARKETING STRATEGY")
    print("="*80)
    print("Based on Population-Weighted Occupation Analysis")
    
    total_population = occ_df['population_represented'].sum()
    total_high_income_pop = occ_df['high_income_population'].sum()
    
    print(f"\n💡 MARKET OVERVIEW:")
    print(f"   • Total Addressable Market: {total_population:,.0f} people")
    print(f"   • High-Income Market: {total_high_income_pop:,.0f} people ({total_high_income_pop/total_population:.1%})")
    print(f"   • Market Segments: {optimal_k} distinct customer segments")
    
    # Rank segments by business priority
    segment_priority = []
    for cluster_id in range(optimal_k):
        info = segment_analysis[cluster_id]
        # Priority score = (high income population) × (income rate) × (market share)
        priority_score = (info['high_income_population'] * info['avg_income_rate'] * info['market_share'])
        segment_priority.append((cluster_id, priority_score, info))
    
    segment_priority.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🎯 SEGMENT PRIORITIZATION & STRATEGY:")
    
    for rank, (cluster_id, priority_score, info) in enumerate(segment_priority, 1):
        print(f"\n{'='*60}")
        print(f"PRIORITY {rank}: MARKET SEGMENT {cluster_id}")
        print(f"{'='*60}")
        
        # Determine segment characteristics
        if info['avg_income_rate'] > 0.3 and info['population'] > total_population * 0.15:
            segment_type = "PREMIUM HIGH-VALUE"
            strategy_focus = "Premium Products & Services"
        elif info['avg_income_rate'] > 0.2:
            segment_type = "AFFLUENT PROFESSIONAL"  
            strategy_focus = "Quality & Convenience"
        elif info['population'] > total_population * 0.25:
            segment_type = "MASS MARKET"
            strategy_focus = "Value & Accessibility"
        else:
            segment_type = "NICHE/SPECIALTY"
            strategy_focus = "Targeted Solutions"
        
        print(f"🏷️  Segment Type: {segment_type}")
        print(f"📊 Market Metrics:")
        print(f"   • Population: {info['population']:,.0f} ({info['market_share']:.1%} of total)")
        print(f"   • High-Income Population: {info['high_income_population']:,.0f}")
        print(f"   • Revenue Potential: ${info['high_income_population'] * 2000:,.0f}+ annually*")
        print(f"   • Average Age: {info['avg_weighted_age']:.1f} years")
        print(f"   • Work Stability: {info['avg_weighted_weeks']:.1f} weeks/year")
        
        print(f"\n🛍️  {strategy_focus.upper()} STRATEGY:")
        
        if segment_type == "PREMIUM HIGH-VALUE":
            print("   • Product Mix: Luxury goods, premium brands, exclusive items")
            print("   • Pricing: Premium pricing, limited-time exclusive offers")
            print("   • Channels: Upscale retail, personal shopping, premium online")
            print("   • Marketing: Quality, status, exclusivity, time-saving")
            print("   • Budget Allocation: 25-30% of marketing spend")
            
        elif segment_type == "AFFLUENT PROFESSIONAL":
            print("   • Product Mix: Professional services, quality brands, convenience")
            print("   • Pricing: Competitive premium, value bundles")
            print("   • Channels: Business districts, online, professional networks")
            print("   • Marketing: Professional growth, efficiency, quality of life")
            print("   • Budget Allocation: 20-25% of marketing spend")
            
        elif segment_type == "MASS MARKET":
            print("   • Product Mix: Popular brands, family products, everyday items")
            print("   • Pricing: Competitive, promotional pricing")
            print("   • Channels: Suburban malls, big box stores, mainstream online")
            print("   • Marketing: Family values, practicality, savings")
            print("   • Budget Allocation: 30-40% of marketing spend")
            
        else:  # NICHE/SPECIALTY
            print("   • Product Mix: Specialized products, niche brands")
            print("   • Pricing: Value-focused, targeted promotions")
            print("   • Channels: Specialized retailers, targeted online")
            print("   • Marketing: Specific needs, community, authenticity")
            print("   • Budget Allocation: 10-15% of marketing spend")
        
        print(f"\n🎯 Key Target Occupations:")
        for occ, row in info['top_occupations'].head(5).iterrows():
            market_value = row['population_represented'] * row['weighted_high_income_rate'] * 1000  # Estimated annual value
            print(f"   • {occ[:35]:<35} Pop: {row['population_represented']:>8,.0f}, Value: ${market_value:>10,.0f}")
    
    print(f"\n🚀 IMPLEMENTATION ROADMAP:")
    print("="*50)
    print("PHASE 1 (0-3 months): Target highest priority segment")
    print(f"   • Focus on Segment {segment_priority[0][0]} ({segment_priority[0][2]['population']:,.0f} people)")
    print("   • Develop premium product lines and marketing campaigns")
    print("   • Establish key partnerships and distribution channels")
    
    print("\nPHASE 2 (3-6 months): Expand to secondary segments")
    print(f"   • Launch campaigns for Segments {segment_priority[1][0]} & {segment_priority[2][0] if len(segment_priority) > 2 else 'N/A'}")
    print("   • Optimize marketing mix based on Phase 1 learnings")
    print("   • Implement cross-segment bundling strategies")
    
    print("\nPHASE 3 (6-12 months): Full market coverage")
    print("   • Complete segment coverage with tailored approaches")
    print("   • Implement data-driven optimization and personalization")
    print("   • Develop loyalty programs and retention strategies")
    
    print(f"\n💰 PROJECTED BUSINESS IMPACT:")
    total_revenue_potential = sum(info['high_income_population'] * info['avg_income_rate'] * 2000 
                                for _, _, info in segment_priority)
    print(f"   • Total Revenue Potential: ${total_revenue_potential:,.0f}+ annually")
    print(f"   • Market Penetration Target: 5-15% depending on segment")
    print(f"   • Expected ROI: 3-5x marketing investment over 12 months")
    
    print(f"\n📊 SUCCESS METRICS:")
    print("   • Market share by segment")
    print("   • Customer acquisition cost (CAC) by occupation group")
    print("   • Customer lifetime value (CLV) by segment")
    print("   • Revenue per segment and occupation")
    print("   • Cross-segment purchase behavior")
    
    return {
        'segment_priority': segment_priority,
        'total_revenue_potential': total_revenue_potential,
        'implementation_phases': 3
    }

def create_enhanced_visualizations(occ_df, clustering_features, optimal_k):
    """Create comprehensive visualizations including population weights."""
    print(f"\n🎨 CREATING ENHANCED VISUALIZATIONS:")
    
    fig = plt.figure(figsize=(24, 18))
    
    # Color palette for clusters
    colors = plt.cm.Set1(np.linspace(0, 1, optimal_k))
    
    # 1. 3D Scatter: Age × Weeks × Income Rate (sized by population)
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    
    for cluster_id in range(optimal_k):
        cluster_data = occ_df[occ_df['cluster'] == cluster_id]
        
        # Size by log population for better visualization
        sizes = np.log10(cluster_data['population_represented']) * 30
        
        ax1.scatter(
            cluster_data['weighted_avg_age'],
            cluster_data['weighted_avg_weeks'],
            cluster_data['weighted_high_income_rate'],
            c=[colors[cluster_id]],
            s=sizes,
            alpha=0.7,
            label=f'Segment {cluster_id}',
            edgecolors='black',
            linewidth=0.5
        )
    
    ax1.set_xlabel('Weighted Avg Age')
    ax1.set_ylabel('Weighted Avg Weeks')
    ax1.set_zlabel('High Income Rate')
    ax1.set_title('3D Market Segmentation\n(Size = Population)', fontweight='bold')
    ax1.legend()
    
    # 2. Population by Segment
    ax2 = fig.add_subplot(3, 3, 2)
    segment_pops = occ_df.groupby('cluster')['population_represented'].sum()
    bars = ax2.bar(range(optimal_k), segment_pops.values, color=colors)
    ax2.set_xlabel('Market Segment')
    ax2.set_ylabel('Population Represented')
    ax2.set_title('Population Size by Segment', fontweight='bold')
    ax2.set_xticks(range(optimal_k))
    ax2.set_xticklabels([f'Segment {i}' for i in range(optimal_k)])
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}', ha='center', va='bottom')
    
    # 3. Income Rate vs Population Size
    ax3 = fig.add_subplot(3, 3, 3)
    for cluster_id in range(optimal_k):
        cluster_data = occ_df[occ_df['cluster'] == cluster_id]
        ax3.scatter(
            cluster_data['population_represented'],
            cluster_data['weighted_high_income_rate'],
            c=[colors[cluster_id]],
            s=80,
            alpha=0.7,
            label=f'Segment {cluster_id}'
        )
    
    ax3.set_xlabel('Population Represented (log scale)')
    ax3.set_ylabel('High Income Rate')
    ax3.set_xscale('log')
    ax3.set_title('Income Rate vs Population Size', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Market Potential Matrix
    ax4 = fig.add_subplot(3, 3, 4)
    market_potential = occ_df['weighted_high_income_rate'] * np.log10(occ_df['population_represented'])
    scatter = ax4.scatter(
        occ_df['weighted_high_income_rate'],
        np.log10(occ_df['population_represented']),
        c=market_potential,
        s=80,
        cmap='viridis',
        alpha=0.7
    )
    ax4.set_xlabel('High Income Rate')
    ax4.set_ylabel('Log Population Size')
    ax4.set_title('Market Potential Matrix', fontweight='bold')
    plt.colorbar(scatter, ax=ax4, label='Market Potential Score')
    
    # 5. Age Distribution by Segment
    ax5 = fig.add_subplot(3, 3, 5)
    for cluster_id in range(optimal_k):
        cluster_data = occ_df[occ_df['cluster'] == cluster_id]
        ax5.hist(cluster_data['weighted_avg_age'], bins=15, alpha=0.6, 
                label=f'Segment {cluster_id}', color=colors[cluster_id])
    
    ax5.set_xlabel('Weighted Average Age')
    ax5.set_ylabel('Number of Occupations')
    ax5.set_title('Age Distribution by Segment', fontweight='bold')
    ax5.legend()
    
    # 6. High Income Population by Segment
    ax6 = fig.add_subplot(3, 3, 6)
    high_income_pops = occ_df.groupby('cluster')['high_income_population'].sum()
    bars = ax6.bar(range(optimal_k), high_income_pops.values, color=colors)
    ax6.set_xlabel('Market Segment')
    ax6.set_ylabel('High Income Population')
    ax6.set_title('High-Income Population by Segment', fontweight='bold')
    ax6.set_xticks(range(optimal_k))
    ax6.set_xticklabels([f'Segment {i}' for i in range(optimal_k)])
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}', ha='center', va='bottom')
    
    # 7. Market Share Pie Chart
    ax7 = fig.add_subplot(3, 3, 7)
    segment_shares = occ_df.groupby('cluster')['population_represented'].sum()
    wedges, texts, autotexts = ax7.pie(segment_shares.values, labels=[f'Segment {i}' for i in range(optimal_k)],
                                      colors=colors, autopct='%1.1f%%', startangle=90)
    ax7.set_title('Market Share by Segment', fontweight='bold')
    
    # 8. Revenue Potential by Segment
    ax8 = fig.add_subplot(3, 3, 8)
    revenue_potential = []
    for cluster_id in range(optimal_k):
        cluster_data = occ_df[occ_df['cluster'] == cluster_id]
        revenue = (cluster_data['high_income_population'] * cluster_data['weighted_high_income_rate'] * 2000).sum()
        revenue_potential.append(revenue)
    
    bars = ax8.bar(range(optimal_k), revenue_potential, color=colors)
    ax8.set_xlabel('Market Segment')
    ax8.set_ylabel('Revenue Potential ($)')
    ax8.set_title('Annual Revenue Potential by Segment', fontweight='bold')
    ax8.set_xticks(range(optimal_k))
    ax8.set_xticklabels([f'Segment {i}' for i in range(optimal_k)])
    
    # Format y-axis as currency
    ax8.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'${height/1e6:.1f}M', ha='center', va='bottom')
    
    # 9. Occupation Count by Segment
    ax9 = fig.add_subplot(3, 3, 9)
    occ_counts = occ_df.groupby('cluster').size()
    bars = ax9.bar(range(optimal_k), occ_counts.values, color=colors)
    ax9.set_xlabel('Market Segment')
    ax9.set_ylabel('Number of Occupations')
    ax9.set_title('Occupation Diversity by Segment', fontweight='bold')
    ax9.set_xticks(range(optimal_k))
    ax9.set_xticklabels([f'Segment {i}' for i in range(optimal_k)])
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('enhanced_weighted_clustering_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Enhanced visualization saved as 'enhanced_weighted_clustering_analysis.png'")
    plt.show()
    
    return fig

def analyze_geographic_and_demographic_patterns(df, occ_df):
    """Analyze geographic and demographic patterns if data is available."""
    print(f"\n🗺️  GEOGRAPHIC & DEMOGRAPHIC PATTERN ANALYSIS:")
    
    # Check for geographic features
    geographic_features = [col for col in df.columns if any(geo in col.lower() 
                          for geo in ['state', 'region', 'msa', 'metro', 'urban', 'rural'])]
    
    demographic_features = [col for col in df.columns if any(demo in col.lower() 
                           for demo in ['sex', 'race', 'marital', 'education', 'veteran', 'disability'])]
    
    patterns = {}
    
    if geographic_features:
        print(f"✓ Found geographic features: {geographic_features}")
        
        for geo_col in geographic_features[:2]:  # Analyze top 2 geographic features
            if geo_col in df.columns:
                print(f"\n📍 {geo_col.upper()} DISTRIBUTION:")
                
                # Calculate weighted distribution by geography
                geo_analysis = []
                for geo_value in df[geo_col].unique():
                    if pd.isna(geo_value):
                        continue
                    
                    geo_data = df[df[geo_col] == geo_value]
                    if len(geo_data) < 100:  # Skip small geographic areas
                        continue
                    
                    total_weight = geo_data['weight'].sum()
                    high_income_weight = geo_data[geo_data['target'] == 1]['weight'].sum()
                    
                    geo_stats = {
                        'geography': geo_value,
                        'population': total_weight,
                        'high_income_population': high_income_weight,
                        'income_rate': high_income_weight / total_weight if total_weight > 0 else 0,
                        'avg_age': np.average(geo_data['age'], weights=geo_data['weight']),
                        'top_occupation': geo_data['major occupation code'].mode().iloc[0] if not geo_data['major occupation code'].mode().empty else 'Unknown'
                    }
                    geo_analysis.append(geo_stats)
                
                geo_df = pd.DataFrame(geo_analysis).sort_values('population', ascending=False)
                
                print(f"{'Area':<20} {'Population':<12} {'Income Rate':<12} {'Avg Age':<8} {'Top Occupation'}")
                print("-" * 80)
                
                for _, row in geo_df.head(10).iterrows():
                    print(f"{str(row['geography'])[:19]:<20} {row['population']:>10,.0f} "
                          f"{row['income_rate']:>10.1%} {row['avg_age']:>7.1f} "
                          f"{str(row['top_occupation'])[:20]}")
                
                patterns[geo_col] = geo_df
    
    if demographic_features:
        print(f"\n👥 DEMOGRAPHIC PATTERN ANALYSIS:")
        
        for demo_col in demographic_features[:3]:  # Analyze top 3 demographic features
            if demo_col in df.columns:
                print(f"\n📊 {demo_col.upper()} BREAKDOWN:")
                
                demo_analysis = df.groupby(demo_col).agg({
                    'weight': 'sum',
                    'target': lambda x: np.average(x, weights=df.loc[x.index, 'weight'])
                }).round(3)
                
                demo_analysis.columns = ['Population', 'Income_Rate']
                demo_analysis = demo_analysis.sort_values('Population', ascending=False)
                
                print(f"{'Category':<20} {'Population':<12} {'Income Rate'}")
                print("-" * 45)
                
                for category, row in demo_analysis.head(10).iterrows():
                    if not pd.isna(category):
                        print(f"{str(category)[:19]:<20} {row['Population']:>10,.0f} {row['Income_Rate']:>10.1%}")
                
                patterns[demo_col] = demo_analysis
    
    return patterns

def generate_actionable_recommendations(occ_df, segment_analysis, patterns, optimal_k):
    """Generate specific, actionable recommendations for the retail client."""
    print(f"\n🎯 ACTIONABLE RECOMMENDATIONS FOR RETAIL CLIENT")
    print("="*70)
    
    total_population = occ_df['population_represented'].sum()
    
    print(f"📋 IMMEDIATE ACTIONS (Next 30 Days):")
    print("-" * 50)
    
    # Find highest value segment
    best_segment_id = max(segment_analysis.keys(), 
                         key=lambda x: segment_analysis[x]['high_income_population'])
    best_segment = segment_analysis[best_segment_id]
    
    print(f"1. 🎯 PRIORITY TARGET: Market Segment {best_segment_id}")
    print(f"   • Market Size: {best_segment['population']:,.0f} people ({best_segment['market_share']:.1%})")
    print(f"   • High-Income Population: {best_segment['high_income_population']:,.0f}")
    print(f"   • Immediate Action: Launch targeted campaign to top 3 occupations")
    
    top_3_occs = best_segment['top_occupations'].head(3)
    total_addressable = top_3_occs['population_represented'].sum()
    
    print(f"\n   🏆 TOP 3 OCCUPATION TARGETS:")
    for i, (occ, row) in enumerate(top_3_occs.iterrows(), 1):
        market_value = row['population_represented'] * row['weighted_high_income_rate'] * 1500
        print(f"   {i}. {occ[:40]}")
        print(f"      Population: {row['population_represented']:,.0f}")
        print(f"      Est. Annual Value: ${market_value:,.0f}")
    
    print(f"\n   💼 IMMEDIATE CAMPAIGN SETUP:")
    print(f"   • Budget Allocation: $50,000-100,000 for initial test")
    print(f"   • Channel Mix: 60% digital, 40% traditional")
    print(f"   • Target Reach: {total_addressable * 0.1:,.0f} people (10% of segment)")
    print(f"   • Expected Response Rate: 2-5% based on income levels")
    
    print(f"\n📊 DATA COLLECTION PRIORITIES:")
    print("-" * 40)
    print("2. 📱 CUSTOMER DATA ENHANCEMENT")
    print("   • Implement occupation tracking in customer profiles")
    print("   • Add income estimation models to CRM")
    print("   • Track geographic distribution of customers")
    print("   • Monitor cross-segment purchase patterns")
    
    print(f"\n🏪 OPERATIONAL RECOMMENDATIONS:")
    print("-" * 40)
    print("3. 📦 INVENTORY & PRODUCT MIX")
    
    # Calculate inventory recommendations by segment
    for segment_id in range(optimal_k):
        segment = segment_analysis[segment_id]
        segment_share = segment['market_share']
        
        if segment['avg_income_rate'] > 0.3:
            product_focus = "Premium/Luxury (30-40% margin)"
            inventory_share = min(segment_share * 1.2, 0.4)
        elif segment['avg_income_rate'] > 0.2:
            product_focus = "Mid-Premium (20-30% margin)"
            inventory_share = segment_share
        else:
            product_focus = "Value/Essential (10-20% margin)"
            inventory_share = max(segment_share * 0.8, 0.2)
        
        print(f"   • Segment {segment_id}: {inventory_share:.1%} inventory → {product_focus}")
    
    print(f"\n🎨 MARKETING CHANNEL STRATEGY:")
    print("-" * 40)
    print("4. 📢 CHANNEL OPTIMIZATION")
    
    # Professional segments
    prof_segments = [seg_id for seg_id, seg in segment_analysis.items() 
                    if seg['avg_weighted_weeks'] > 45 and seg['avg_income_rate'] > 0.25]
    
    if prof_segments:
        prof_population = sum(segment_analysis[seg]['population'] for seg in prof_segments)
        print(f"   • LinkedIn/Professional (Segments {prof_segments}): ${prof_population*0.05:,.0f} budget")
        print(f"     Target: {prof_population:,.0f} working professionals")
    
    # Mass market segments  
    mass_segments = [seg_id for seg_id, seg in segment_analysis.items() 
                    if seg['market_share'] > 0.2]
    
    if mass_segments:
        mass_population = sum(segment_analysis[seg]['population'] for seg in mass_segments)
        print(f"   • Facebook/Instagram (Segments {mass_segments}): ${mass_population*0.03:,.0f} budget")
        print(f"     Target: {mass_population:,.0f} general consumers")
    
    print(f"\n💰 FINANCIAL PROJECTIONS:")
    print("-" * 40)
    print("5. 📈 REVENUE IMPACT (12-Month Projection)")
    
    total_revenue_potential = 0
    for segment_id, segment in segment_analysis.items():
        # Conservative estimate: 2% market penetration, $1,200 average spend
        segment_revenue = segment['high_income_population'] * 0.02 * 1200
        total_revenue_potential += segment_revenue
        
        print(f"   • Segment {segment_id}: ${segment_revenue:,.0f} "
              f"({segment['high_income_population']*0.02:,.0f} customers)")
    
    print(f"\n   🎯 TOTAL REVENUE POTENTIAL: ${total_revenue_potential:,.0f}")
    print(f"   💸 Required Marketing Investment: ${total_revenue_potential*0.15:,.0f} (15% of revenue)")
    print(f"   📊 Expected ROI: {total_revenue_potential/(total_revenue_potential*0.15):.1f}x")
    
    print(f"\n🚨 RISK MITIGATION:")
    print("-" * 30)
    print("6. ⚠️  KEY RISKS & MITIGATION")
    print("   • Market Saturation: Start with underserved segments")
    print("   • Economic Downturn: Focus on essential product categories")
    print("   • Competition: Develop unique value propositions per segment")
    print("   • Data Privacy: Ensure compliant data collection methods")
    
    print(f"\n✅ SUCCESS METRICS & KPIs:")
    print("-" * 30)
    print("7. 📊 MEASUREMENT FRAMEWORK")
    print("   • Customer Acquisition Cost (CAC) by segment: <$50")
    print("   • Customer Lifetime Value (CLV) by occupation: >$1,000")
    print("   • Market penetration by segment: 2-5% in Year 1")
    print("   • Revenue per segment: Track monthly growth")
    print("   • Cross-segment purchase rate: Target 15%+")
    
    return {
        'priority_segment': best_segment_id,
        'total_revenue_potential': total_revenue_potential,
        'marketing_investment': total_revenue_potential * 0.15,
        'top_occupations': top_3_occs.index.tolist()
    }

def main():
    """Run the complete enhanced weighted clustering analysis."""
    print("STARTING ENHANCED WEIGHTED CLUSTERING ANALYSIS")
    print("="*80)
    
    # Load weighted data
    df = load_and_prepare_weighted_data()
    
    # Calculate weighted occupation features
    occ_df = calculate_weighted_occupation_features(df)
    
    # Perform weighted clustering
    clustering_features, occ_df, scaler, kmeans, optimal_k = perform_weighted_clustering_analysis(occ_df)
    
    # Analyze market segments
    segment_analysis = analyze_weighted_market_segments(occ_df, optimal_k)
    
    # Create enhanced visualizations
    fig = create_enhanced_visualizations(occ_df, clustering_features, optimal_k)
    
    # Analyze geographic and demographic patterns
    patterns = analyze_geographic_and_demographic_patterns(df, occ_df)
    
    # Generate marketing strategy
    marketing_strategy = generate_enhanced_marketing_strategy(occ_df, segment_analysis, optimal_k)
    
    # Generate actionable recommendations
    recommendations = generate_actionable_recommendations(occ_df, segment_analysis, patterns, optimal_k)
    
    # Save comprehensive results
    occ_df.to_csv('weighted_occupation_analysis.csv')
    
    # Create summary report
    summary_df = pd.DataFrame([
        {
            'segment_id': seg_id,
            'population': info['population'],
            'market_share': info['market_share'],
            'high_income_population': info['high_income_population'],
            'avg_income_rate': info['avg_income_rate'],
            'avg_age': info['avg_weighted_age'],
            'occupations_count': info['occupations'],
            'revenue_potential': info['high_income_population'] * info['avg_income_rate'] * 2000
        }
        for seg_id, info in segment_analysis.items()
    ])
    
    summary_df.to_csv('market_segments_summary.csv', index=False)
    
    print(f"\n💾 RESULTS SAVED:")
    print("   • weighted_occupation_analysis.csv (detailed occupation data)")
    print("   • market_segments_summary.csv (segment summary)")
    print("   • enhanced_weighted_clustering_analysis.png (visualizations)")
    
    print(f"\n" + "="*80)
    print("ENHANCED WEIGHTED CLUSTERING ANALYSIS COMPLETE")
    print("="*80)
    print(f"✓ Population-weighted analysis of {len(occ_df)} occupations")
    print(f"✓ {optimal_k} market segments identified with ${marketing_strategy['total_revenue_potential']:,.0f} potential")
    print(f"✓ Geographic and demographic patterns analyzed")
    print(f"✓ Comprehensive marketing strategy and actionable recommendations generated")
    print(f"✓ ROI projection: {marketing_strategy['total_revenue_potential']/recommendations['marketing_investment']:.1f}x return")
    
    return {
        'occupation_data': occ_df,
        'segment_analysis': segment_analysis,
        'marketing_strategy': marketing_strategy,
        'recommendations': recommendations,
        'patterns': patterns,
        'models': {'scaler': scaler, 'kmeans': kmeans}
    }

if __name__ == "__main__":
    results = main()