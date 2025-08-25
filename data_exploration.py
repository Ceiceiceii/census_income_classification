import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from scipy.stats import chi2_contingency
import warnings
import os
warnings.filterwarnings('ignore')

# Create eda_plots directory if it doesn't exist
os.makedirs('eda_plots', exist_ok=True)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_basic_exploration():
    print("="*60)
    print("CENSUS DATA EXPLORATION")
    print("="*60)
    
    df = pd.read_csv('census_data.csv')
    print(f"Dataset Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nData Types:")
    print(df.dtypes.value_counts())
    
    # missing values analysis
    print(f"\nMissing Values Analysis:")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    if len(missing_data) > 0:
        print(missing_data)
    else:
        print("No missing values found")
    
    # check for '?' values
    print(f"\nChecking for '?' values (missing indicators):")
    question_mark_counts = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            count = (df[col] == '?').sum()
            if count > 0:
                question_mark_counts[col] = count
    
    if question_mark_counts:
        for col, count in sorted(question_mark_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
    
    return df

def analyze_target_variable(df):
    print(f"\n" + "="*60)
    print("TARGET VARIABLE ANALYSIS")
    print("="*60)
    
    # binary target
    df['target'] = (~df['label'].str.contains('- 50000', na=False)).astype(int)
    
    target_counts = df['target'].value_counts()
    print(f"Target Distribution:")
    print(f"  ≤$50K (0): {target_counts[0]:,} ({target_counts[0]/len(df)*100:.2f}%)")
    print(f"  >$50K (1): {target_counts[1]:,} ({target_counts[1]/len(df)*100:.2f}%)")
    
    imbalance_ratio = target_counts[0] / target_counts[1]
    print(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    # visualize target distribution
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    target_counts.plot(kind='bar', color=['skyblue', 'lightcoral'])
    plt.title('Income Distribution (Counts)')
    plt.xlabel('Income Level')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['≤$50K', '>$50K'], rotation=0)
    
    plt.subplot(1, 2, 2)
    plt.pie(target_counts.values, labels=['≤$50K', '>$50K'], autopct='%1.1f%%', 
            colors=['skyblue', 'lightcoral'])
    plt.title('Income Distribution (Percentage)')
    
    plt.tight_layout()
    plt.savefig('eda_plots/target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def analyze_categorical_features(df):
    print(f"\n" + "="*60)
    print("CATEGORICAL FEATURES ANALYSIS")
    print("="*60)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != 'label']
    
    print(f"Found {len(categorical_cols)} categorical features")
    
    # analyze each categorical feature
    categorical_analysis = {}
    
    for col in categorical_cols:
        unique_values = df[col].nunique()
        most_common = df[col].value_counts().head(5)
        
        categorical_analysis[col] = {
            'unique_count': unique_values,
            'most_common': most_common
        }
        
        print(f"\n{col.upper()}:")
        print(f"  Unique values: {unique_values}")
        print(f"  Top 5 values:")
        for val, count in most_common.items():
            print(f"    {val}: {count:,} ({count/len(df)*100:.2f}%)")
    
    # visualize categorical features
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    key_features = ['education', 'marital stat', 'race', 'sex']
    
    for i, col in enumerate(key_features):
        if col in df.columns:
            crosstab = pd.crosstab(df[col], df['target'], normalize='index')
            crosstab.plot(kind='bar', ax=axes[i], stacked=False, 
                         color=['skyblue', 'lightcoral'])
            axes[i].set_title(f'Income Distribution by {col.title()}')
            axes[i].set_xlabel(col.title())
            axes[i].set_ylabel('Proportion')
            axes[i].legend(['≤$50K', '>$50K'])
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('eda_plots/categorical_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return categorical_analysis

def analyze_numerical_features(df):
    print(f"\n" + "="*60)
    print("NUMERICAL FEATURES ANALYSIS")
    print("="*60)
    
    #numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in ['target']]
    
    print(f"Found {len(numerical_cols)} numerical features:")
    for col in numerical_cols:
        print(f"  - {col}")
    
    print(f"\nNumerical Features Statistics:")
    numerical_stats = df[numerical_cols].describe()
    print(numerical_stats)
    
    # Check for outlier
    print(f"\nOutlier Analysis (IQR method):")
    outlier_info = {}
    
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(df)) * 100
        
        outlier_info[col] = {
            'count': outlier_count,
            'percentage': outlier_percentage,
            'bounds': (lower_bound, upper_bound)
        }
        
        if outlier_percentage > 5:  
            #significant outliers
            print(f"  {col}: {outlier_count:,} outliers ({outlier_percentage:.2f}%)")
    
    # visualize distributions
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.ravel()
    
    for i, col in enumerate(numerical_cols[:9]):
        # Distribution by target
        high_income = df[df['target'] == 1][col]
        low_income = df[df['target'] == 0][col]
        
        axes[i].hist(low_income, bins=50, alpha=0.7, label='≤$50K', color='skyblue', density=True)
        axes[i].hist(high_income, bins=50, alpha=0.7, label='>$50K', color='lightcoral', density=True)
        axes[i].set_title(f'{col} Distribution by Income')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Density')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('eda_plots/numerical_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return numerical_stats, outlier_info

def analyze_feature_relationships(df):
    print(f"\n" + "="*60)
    print("FEATURE RELATIONSHIP ANALYSIS")
    print("="*60)
    
    # Encode categorical variables
    df_encoded = df.copy()
    le = LabelEncoder()
    
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != 'label']
    
    for col in categorical_cols:
        df_encoded[col] = df_encoded[col].fillna('Unknown')
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    # Calculate correlation matrix
    correlation_matrix = df_encoded.drop(['label'], axis=1).corr()
    
    # Find highly correlated feature pairs
    print("Highly Correlated Feature Pairs (|correlation| > 0.7):")
    high_corr_pairs = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:
                high_corr_pairs.append({
                    'feature1': correlation_matrix.columns[i],
                    'feature2': correlation_matrix.columns[j],
                    'correlation': corr_value
                })
                print(f"  {correlation_matrix.columns[i]} <-> {correlation_matrix.columns[j]}: {corr_value:.3f}")
    
    if not high_corr_pairs:
        print("  No highly correlated pairs found (threshold: 0.7)")
    
    # visualize correlation matrix
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', 
                center=0, square=True, linewidths=0.1, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('eda_plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Correlation with target variable
    target_correlations = correlation_matrix['target'].abs().sort_values(ascending=False)
    target_correlations = target_correlations.drop('target')
    
    print(f"\nTop 15 Features Most Correlated with Target:")
    for feature, corr in target_correlations.head(15).items():
        print(f"  {feature}: {corr:.4f}")
    
    # Visualize top correlations with target
    plt.figure(figsize=(12, 8))
    top_15_corr = target_correlations.head(15)
    plt.barh(range(len(top_15_corr)), top_15_corr.values, color='lightblue')
    plt.yticks(range(len(top_15_corr)), top_15_corr.index)
    plt.xlabel('Absolute Correlation with Target')
    plt.title('Top 15 Features Most Correlated with Income Level')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('eda_plots/target_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return high_corr_pairs, target_correlations

def feature_importance_analysis(df):
    print(f"\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # prep data
    df_encoded = df.copy()
    le = LabelEncoder()
    
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != 'label']
    
    for col in categorical_cols:
        df_encoded[col] = df_encoded[col].fillna('Unknown')
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    X = df_encoded.drop(['label', 'target'], axis=1)
    y = df_encoded['target']
    
    # Chi-square test
    chi2_scores, chi2_pvals = chi2(X, y)
    chi2_results = pd.DataFrame({
        'feature': X.columns,
        'chi2_score': chi2_scores,
        'chi2_pval': chi2_pvals
    }).sort_values('chi2_score', ascending=False)
    
    print("Top 15 Features by Chi-square Score:")
    for _, row in chi2_results.head(15).iterrows():
        print(f"  {row['feature']}: {row['chi2_score']:.2f} (p-value: {row['chi2_pval']:.2e})")
    
    # F-stat numerical features
    f_scores, f_pvals = f_classif(X, y)
    f_results = pd.DataFrame({
        'feature': X.columns,
        'f_score': f_scores,
        'f_pval': f_pvals
    }).sort_values('f_score', ascending=False)
    
    print(f"\nTop 15 Features by F-statistic:")
    for _, row in f_results.head(15).iterrows():
        print(f"  {row['feature']}: {row['f_score']:.2f} (p-value: {row['f_pval']:.2e})")
    
    # mutual information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_results = pd.DataFrame({
        'feature': X.columns,
        'mutual_info': mi_scores
    }).sort_values('mutual_info', ascending=False)
    
    print(f"\nTop 15 Features by Mutual Information:")
    for _, row in mi_results.head(15).iterrows():
        print(f"  {row['feature']}: {row['mutual_info']:.4f}")
    
    return chi2_results, f_results, mi_results

def main():
    df = load_and_basic_exploration()
    
    df = analyze_target_variable(df)
    
    categorical_analysis = analyze_categorical_features(df)
    
    numerical_stats, outlier_info = analyze_numerical_features(df)
    
    high_corr_pairs, target_correlations = analyze_feature_relationships(df)
    
    chi2_results, f_results, mi_results = feature_importance_analysis(df)
    
    print(f"\n" + "="*60)
    print("DATA INSIGHTS & RECOMMENDATIONS")
    print("="*60)
    
    print("DATASET CHARACTERISTICS:")
    print(f" - {df.shape[0]:,} samples with {df.shape[1]-2} features")
    print(f" - Highly imbalanced: {(df['target']==0).sum():,} low-income vs {(df['target']==1).sum():,} high-income")
    print(f" - Mix of categorical ({len(categorical_analysis)}) and numerical features")
    
    print(f"KEY FINDINGS:")
    if high_corr_pairs:
        print(f"  - {len(high_corr_pairs)} highly correlated feature pairs found")
    
    print(f"  - Top predictive features: {', '.join(target_correlations.head(5).index)}")

if __name__ == "__main__":
    main()