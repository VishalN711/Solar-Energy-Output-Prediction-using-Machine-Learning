#!/usr/bin/env python3
"""
================================================================================
SOLAR ENERGY PREDICTION - MACHINE LEARNING TRAINING & EVALUATION
================================================================================
Complete ML Pipeline: Data Loading, Splitting, Training, Prediction, Evaluation

Features:
- Loads preprocessed solar energy dataset (Part 1 & 2)
- Splits data into training (80%) and testing (20%)
- Trains 5 different ML models
- Makes predictions on test set
- Compares predictions with actual values
- Calculates accuracy metrics (R², RMSE, MAE, MAPE)
- Saves results to CSV files
- Generates performance visualizations

Author: AI Research Assistant
Date: November 2025
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load preprocessed datasets from CSV files."""
    print("\n" + "="*80)
    print("[1] LOADING PREPROCESSED DATASETS")
    print("="*80)
    
    try:
        df_part1 = pd.read_csv('preprocessed_solar_data_part1.csv')
        df_part2 = pd.read_csv('preprocessed_solar_data_part2.csv')
        df = pd.concat([df_part1, df_part2], axis=0, ignore_index=True)
        
        print(f"✓ Part 1 records: {len(df_part1)}")
        print(f"✓ Part 2 records: {len(df_part2)}")
        print(f"✓ Total records: {len(df)}")
        print(f"✓ Total features: {df.shape[1]}")
        print("✓ Data loaded successfully!")
        
        return df
    except FileNotFoundError:
        print(" Error: CSV files not found!")
        print("   Please ensure these files exist:")
        print("   - preprocessed_solar_data_part1.csv")
        print("   - preprocessed_solar_data_part2.csv")
        exit(1)


def prepare_data(df):
    """Prepare data for machine learning."""
    print("\n" + "="*80)
    print("[2] PREPARING DATA FOR MACHINE LEARNING")
    print("="*80)
    
    # Drop non-numeric columns
    df_model = df.drop(['Date', 'Season'], axis=1)
    
    # Separate features and target
    X = df_model.drop('Solar_Energy_Output', axis=1)
    y = df_model['Solar_Energy_Output']
    
    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    print(f"✓ Feature columns: {list(X.columns)}")
    print("✓ Data prepared successfully!")
    
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    print("\n" + "="*80)
    print("[3] SPLITTING DATA INTO TRAIN AND TEST SETS")
    print("="*80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"✓ Training set: {len(X_train)} records ({len(X_train)/len(X)*100:.1f}%)")
    print(f"✓ Testing set: {len(X_test)} records ({len(X_test)/len(X)*100:.1f}%)")
    print(f"✓ Training features shape: {X_train.shape}")
    print(f"✓ Testing features shape: {X_test.shape}")
    print("✓ Data split completed!")
    
    return X_train, X_test, y_train, y_test


def initialize_models():
    """Initialize machine learning models."""
    print("\n" + "="*80)
    print("[4] INITIALIZING MACHINE LEARNING MODELS")
    print("="*80)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
        'Support Vector Machine': SVR(kernel='rbf', C=1.0, epsilon=0.1)
    }
    
    print(f"✓ Number of models: {len(models)}")
    for name in models.keys():
        print(f"  • {name}")
    print("✓ Models initialized!")
    
    return models


def train_models(models, X_train, y_train):
    """Train all machine learning models."""
    print("\n" + "="*80)
    print("[5] TRAINING MACHINE LEARNING MODELS")
    print("="*80)
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...", end=" ")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print("✓")
    
    print("✓ All models trained successfully!")
    
    return trained_models


def make_predictions(trained_models, X_test):
    """Make predictions on test set."""
    print("\n" + "="*80)
    print("[6] MAKING PREDICTIONS ON TEST SET")
    print("="*80)
    
    predictions = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        print(f"✓ {name}: {len(y_pred)} predictions made")
    
    print("✓ All predictions completed!")
    
    return predictions


def evaluate_models(trained_models, predictions, X_train, X_test, y_train, y_test):
    """Evaluate all models and calculate metrics."""
    print("\n" + "="*80)
    print("[7] EVALUATING MODEL PERFORMANCE")
    print("="*80)
    
    results = {}
    
    print(f"\n{'Model':<25} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R² Score':<12} {'Accuracy%':<12}")
    print("-" * 95)
    
    for name, model in trained_models.items():
        # Get training and testing predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = predictions[name]
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        accuracy = r2 * 100
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_test - y_test_pred) / (y_test + 1e-10))) * 100
        
        # Store results
        results[name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'accuracy': accuracy,
            'mape': mape,
            'train_r2': r2_score(y_train, y_train_pred),
            'predictions': y_test_pred
        }
        
        print(f"{name:<25} {mse:<12.4f} {rmse:<12.4f} {mae:<12.4f} {r2:<12.4f} {accuracy:<12.2f}")
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['r2_score'])
    best_r2 = results[best_model_name]['r2_score']
    
    print("-" * 95)
    print(f"\n BEST MODEL: {best_model_name} (R² Score: {best_r2:.4f})")
    print("✓ Evaluation completed!")
    
    return results, best_model_name


def compare_predictions_with_actual(results, best_model_name, y_test):
    """Compare predictions with actual values."""
    print("\n" + "="*80)
    print("[8] COMPARISON OF PREDICTIONS WITH ACTUAL VALUES")
    print("="*80)
    
    comparison_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': results[best_model_name]['predictions'],
        'Difference': y_test.values - results[best_model_name]['predictions'],
        'Absolute_Error': np.abs(y_test.values - results[best_model_name]['predictions']),
        'Percentage_Error': np.abs((y_test.values - results[best_model_name]['predictions']) / (y_test.values + 1e-10) * 100)
    })
    
    print(f"\nBest Model: {best_model_name}")
    print(f"\nFirst 10 predictions vs actual values:")
    print(comparison_df[['Actual', 'Predicted', 'Absolute_Error']].head(10).to_string(index=False))
    
    print(f"\n\nStatistics of Prediction Errors:")
    print(f"  Mean Absolute Error: {comparison_df['Absolute_Error'].mean():.4f}")
    print(f"  Max Absolute Error: {comparison_df['Absolute_Error'].max():.4f}")
    print(f"  Min Absolute Error: {comparison_df['Absolute_Error'].min():.4f}")
    print(f"  Std of Errors: {comparison_df['Difference'].std():.4f}")
    
    return comparison_df


def check_accuracy(results):
    """Display accuracy summary for all models."""
    print("\n" + "="*80)
    print("[9] ACCURACY SUMMARY FOR ALL MODELS")
    print("="*80)
    
    accuracy_summary = []
    for name, metrics in results.items():
        accuracy_summary.append({
            'Model': name,
            'R² Score': f"{metrics['r2_score']:.4f}",
            'Accuracy (%)': f"{metrics['accuracy']:.2f}%",
            'RMSE': f"{metrics['rmse']:.4f}",
            'MAE': f"{metrics['mae']:.4f}",
            'MAPE (%)': f"{metrics['mape']:.2f}%"
        })
    
    df_accuracy = pd.DataFrame(accuracy_summary)
    print("\n" + df_accuracy.to_string(index=False))


def feature_importance_analysis(trained_models, X):
    """Analyze feature importance from Random Forest model."""
    print("\n" + "="*80)
    print("[10] FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    if 'Random Forest' in trained_models:
        rf_model = trained_models['Random Forest']
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features (Random Forest):")
        print(feature_importance.head(10).to_string(index=False))
        
        return feature_importance
    
    return None


def save_results(comparison_df, results):
    """Save results to CSV files."""
    print("\n" + "="*80)
    print("[11] SAVING RESULTS TO FILES")
    print("="*80)
    
    # Save detailed comparison
    comparison_df.to_csv('predictions_vs_actual.csv', index=False)
    print("✓ Saved: predictions_vs_actual.csv")
    
    # Save model performance summary
    results_summary = pd.DataFrame({
        'Model': list(results.keys()),
        'MSE': [results[m]['mse'] for m in results.keys()],
        'RMSE': [results[m]['rmse'] for m in results.keys()],
        'MAE': [results[m]['mae'] for m in results.keys()],
        'R2_Score': [results[m]['r2_score'] for m in results.keys()],
        'Accuracy_Percent': [results[m]['accuracy'] for m in results.keys()],
        'MAPE_Percent': [results[m]['mape'] for m in results.keys()],
        'Train_R2': [results[m]['train_r2'] for m in results.keys()]
    }).sort_values('R2_Score', ascending=False)
    
    results_summary.to_csv('model_performance_summary.csv', index=False)
    print("✓ Saved: model_performance_summary.csv")


def generate_visualizations(results, best_model_name, y_test, X, feature_importance=None):
    """Generate visualization plots."""
    print("\n" + "="*80)
    print("[12] GENERATING VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Model Performance Comparison (R² Score)
    model_names = list(results.keys())
    r2_scores = [results[m]['r2_score'] for m in model_names]
    colors = ['green' if name == best_model_name else 'skyblue' for name in model_names]
    
    axes[0, 0].barh(model_names, r2_scores, color=colors)
    axes[0, 0].set_xlabel('R² Score')
    axes[0, 0].set_title('Model Comparison - R² Score')
    axes[0, 0].set_xlim([0, 1])
    axes[0, 0].grid(axis='x', alpha=0.3)
    for i, v in enumerate(r2_scores):
        axes[0, 0].text(v + 0.02, i, f'{v:.4f}', va='center')
    
    # Plot 2: RMSE Comparison
    rmse_scores = [results[m]['rmse'] for m in model_names]
    axes[0, 1].bar(range(len(model_names)), rmse_scores, color=colors)
    axes[0, 1].set_xticks(range(len(model_names)))
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('Model Comparison - RMSE (Lower is Better)')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Predictions vs Actual (Best Model)
    y_pred_best = results[best_model_name]['predictions']
    axes[1, 0].scatter(y_test, y_pred_best, alpha=0.6, edgecolors='k', s=50)
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                    'r--', lw=2, label='Perfect Prediction')
    axes[1, 0].set_xlabel('Actual Values')
    axes[1, 0].set_ylabel('Predicted Values')
    axes[1, 0].set_title(f'{best_model_name} - Predictions vs Actual')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Residual Plot (Best Model)
    residuals = y_test.values - y_pred_best
    axes[1, 1].scatter(y_pred_best, residuals, alpha=0.6, edgecolors='k', s=50)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title(f'{best_model_name} - Residual Plot')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_evaluation_plots.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: model_evaluation_plots.png")
    
    # Feature Importance Plot
    if feature_importance is not None:
        plt.figure(figsize=(10, 6))
        top_features = feature_importance.head(10)
        plt.barh(top_features['Feature'], top_features['Importance'], color='teal')
        plt.xlabel('Importance')
        plt.title('Top 10 Feature Importance (Random Forest)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: feature_importance.png")
    
    plt.close('all')


def print_final_summary(results, best_model_name, df, X_train, X_test):
    """Print final summary."""
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print(f"\n📊 Dataset Information:")
    print(f"  • Total records: {len(df)}")
    print(f"  • Training records: {len(X_train)}")
    print(f"  • Testing records: {len(X_test)}")
    print(f"  • Number of features: {X_train.shape[1]}")
    
    print(f"\n🤖 Models Trained:")
    print(f"  • Number of models: {len(results)}")
    print(f"  • Models: {', '.join(results.keys())}")
    
    print(f"\n🏆 Best Performing Model:")
    print(f"  • Model: {best_model_name}")
    print(f"  • R² Score: {results[best_model_name]['r2_score']:.4f}")
    print(f"  • Accuracy: {results[best_model_name]['accuracy']:.2f}%")
    print(f"  • RMSE: {results[best_model_name]['rmse']:.4f}")
    print(f"  • MAE: {results[best_model_name]['mae']:.4f}")
    print(f"  • MAPE: {results[best_model_name]['mape']:.2f}%")
    
    print(f"\n📁 Files Generated:")
    print(f"  • predictions_vs_actual.csv")
    print(f"  • model_performance_summary.csv")
    print(f"  • model_evaluation_plots.png")
    print(f"  • feature_importance.png")
    
    print(f"\n✅ MACHINE LEARNING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("SOLAR ENERGY PREDICTION - MACHINE LEARNING PIPELINE")
    print("="*80)
    
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Prepare data
    X, y = prepare_data(df)
    
    # Step 3: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 4: Initialize models
    models = initialize_models()
    
    # Step 5: Train models
    trained_models = train_models(models, X_train, y_train)
    
    # Step 6: Make predictions
    predictions = make_predictions(trained_models, X_test)
    
    # Step 7: Evaluate models
    results, best_model_name = evaluate_models(trained_models, predictions, X_train, X_test, y_train, y_test)
    
    # Step 8: Compare predictions with actual
    comparison_df = compare_predictions_with_actual(results, best_model_name, y_test)
    
    # Step 9: Check accuracy
    check_accuracy(results)
    
    # Step 10: Feature importance
    feature_importance = feature_importance_analysis(trained_models, X)
    
    # Step 11: Save results
    save_results(comparison_df, results)
    
    # Step 12: Generate visualizations
    generate_visualizations(results, best_model_name, y_test, X, feature_importance)
    
    # Print final summary
    print_final_summary(results, best_model_name, df, X_train, X_test)


if __name__ == "__main__":
    main()
