import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ============================================================
# 1) Synthetic data generation
# ============================================================

MSP_PRICE = {
    "rice": 2200,
    "wheat": 2100,
    "maize": 1850,
    "cotton": 6200,
    "soybean": 4700,
    "mustard": 5000,
}

def generate_farm_data(n_rows: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    crop_choices = list(MSP_PRICE.keys())
    soil_choices = ["loam", "clay", "sandy"]

    rows = []
    for _ in range(n_rows):
        crop = rng.choice(crop_choices)
        soil = rng.choice(soil_choices)
        area_ha = rng.uniform(0.5, 8.0)

        rainfall_mm = rng.uniform(300, 1200)
        temperature_c = rng.uniform(18, 40)
        nitrogen_kg_ha = rng.uniform(40, 180)
        phosphorus_kg_ha = rng.uniform(20, 100)
        potassium_kg_ha = rng.uniform(20, 150)

        irrigation_liters = rng.uniform(1500, 10000)
        fertilizer_kg = rng.uniform(20, 220)
        energy_kwh = rng.uniform(50, 500)

        # Soil moisture is influenced by rainfall, irrigation, soil type, and crop
        soil_moisture = (
            60
            + (rainfall_mm / 20)
            + (irrigation_liters / 300)
            - (temperature_c * 0.8)
            - (0.7 if soil == "sandy" else 0.2 if soil == "loam" else 0.5)
        )
        soil_moisture = np.clip(soil_moisture, 15, 95)

        # Crop health is influenced by water, fertility and temperature
        crop_health_score = (
            60
            + (soil_moisture - 40) * 0.9
            + (nitrogen_kg_ha - 80) * 0.12
            + (phosphorus_kg_ha - 50) * 0.06
            + (potassium_kg_ha - 50) * 0.08
            - (temperature_c - 28) * 0.9
        )
        crop_health_score = np.clip(crop_health_score, 0, 100)

        # Yield depends on crop, health, soil moisture, nutrients and water use efficiency
        yield_tons = (
            {
                "rice": 3.5,
                "wheat": 2.4,
                "maize": 2.8,
                "cotton": 1.5,
                "soybean": 1.8,
                "mustard": 1.2,
            }[crop]
            + (crop_health_score / 25)
            + (soil_moisture / 25)
            + (nitrogen_kg_ha / 100) * 0.8
            - (temperature_c / 35) * 0.7
            + rng.uniform(-0.5, 0.8)
        )

        yield_tons = max(yield_tons, 0.1)

        msp = MSP_PRICE[crop]
        income_rs = yield_tons * area_ha * msp

        rows.append({
            "field_id": f"F-{len(rows)+1:03d}",
            "crop": crop,
            "soil_type": soil,
            "area_ha": round(area_ha, 2),
            "rainfall_mm": round(rainfall_mm, 2),
            "temperature_c": round(temperature_c, 2),
            "nitrogen_kg_ha": round(nitrogen_kg_ha, 2),
            "phosphorus_kg_ha": round(phosphorus_kg_ha, 2),
            "potassium_kg_ha": round(potassium_kg_ha, 2),
            "irrigation_liters": round(irrigation_liters, 2),
            "fertilizer_kg": round(fertilizer_kg, 2),
            "energy_kwh": round(energy_kwh, 2),
            "soil_moisture_pct": round(soil_moisture, 2),
            "crop_health_score": round(crop_health_score, 2),
            "yield_tons": round(yield_tons, 2),
            "msp_price_rs_per_ton": msp,
            "income_rs": round(income_rs, 2),
        })

    return pd.DataFrame(rows)


# ============================================================
# 2) ML models
# ============================================================

def make_regression_pipeline():
    categorical_features = ["crop", "soil_type"]
    numerical_features = [
        "area_ha",
        "rainfall_mm",
        "temperature_c",
        "nitrogen_kg_ha",
        "phosphorus_kg_ha",
        "potassium_kg_ha",
        "irrigation_liters",
        "fertilizer_kg",
        "energy_kwh",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=None,
        min_samples_leaf=2,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def train_models(df: pd.DataFrame):
    # Targets
    X = df[
        [
            "crop",
            "soil_type",
            "area_ha",
            "rainfall_mm",
            "temperature_c",
            "nitrogen_kg_ha",
            "phosphorus_kg_ha",
            "potassium_kg_ha",
            "irrigation_liters",
            "fertilizer_kg",
            "energy_kwh",
        ]
    ]

    moisture_model = make_regression_pipeline()
    health_model = make_regression_pipeline()
    yield_model = make_regression_pipeline()

    moisture_model.fit(X, df["soil_moisture_pct"])
    health_model.fit(X, df["crop_health_score"])
    yield_model.fit(X, df["yield_tons"])

    return {
        "soil_moisture_model": moisture_model,
        "crop_health_model": health_model,
        "yield_model": yield_model,
    }


# ============================================================
# 3) Farm digital profile
# ============================================================

@dataclass
class FarmField:
    field_id: str
    crop: str
    soil_type: str
    area_ha: float
    rainfall_mm: float
    temperature_c: float
    nitrogen_kg_ha: float
    phosphorus_kg_ha: float
    potassium_kg_ha: float
    irrigation_liters: float
    fertilizer_kg: float
    energy_kwh: float
    msp_price_rs_per_ton: float

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


def get_farm_profile(field: FarmField) -> Dict[str, Any]:
    return {
        "field_id": field.field_id,
        "crop": field.crop,
        "soil_type": field.soil_type,
        "area_ha": field.area_ha,
        "rainfall_mm": field.rainfall_mm,
        "temperature_c": field.temperature_c,
        "nitrogen_kg_ha": field.nitrogen_kg_ha,
        "phosphorus_kg_ha": field.phosphorus_kg_ha,
        "potassium_kg_ha": field.potassium_kg_ha,
        "irrigation_liters": field.irrigation_liters,
        "fertilizer_kg": field.fertilizer_kg,
        "energy_kwh": field.energy_kwh,
        "msp_price_rs_per_ton": field.msp_price_rs_per_ton,
    }


# ============================================================
# 4) Farm health dashboard
# ============================================================

def farm_health_dashboard(df: pd.DataFrame) -> Dict[str, Any]:
    summary = {
        "fields": int(df["field_id"].nunique()),
        "average_soil_moisture_pct": round(df["soil_moisture_pct"].mean(), 2),
        "average_crop_health_score": round(df["crop_health_score"].mean(), 2),
        "average_yield_tons": round(df["yield_tons"].mean(), 2),
        "total_water_used_liters": round(df["irrigation_liters"].sum(), 2),
        "total_fertilizer_kg": round(df["fertilizer_kg"].sum(), 2),
        "total_energy_kwh": round(df["energy_kwh"].sum(), 2),
        "average_income_rs": round(df["income_rs"].mean(), 2),
    }

    # Crop-wise health status
    crop_summary = (
        df.groupby("crop")
        .agg(
            avg_soil_moisture=("soil_moisture_pct", "mean"),
            avg_health=("crop_health_score", "mean"),
            avg_yield=("yield_tons", "mean"),
            avg_income=("income_rs", "mean"),
        )
        .reset_index()
    )

    return {
        "summary": summary,
        "crop_summary": crop_summary.to_dict(orient="records"),
    }


# ============================================================
# 5) What-if simulation engine
# ============================================================

def simulate_scenario(field: FarmField, scenario: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
    """
    scenario keys example:
    {
      "crop": "rice",
      "soil_type": "loam",
      "irrigation_liters": 6000,
      "fertilizer_kg": 120,
      "rainfall_mm": 950,
      "temperature_c": 30,
      "nitrogen_kg_ha": 120,
      "phosphorus_kg_ha": 60,
      "potassium_kg_ha": 80,
      "energy_kwh": 240
    }
    """
    # baseline values
    baseline = {
        "crop": field.crop,
        "soil_type": field.soil_type,
        "area_ha": field.area_ha,
        "rainfall_mm": field.rainfall_mm,
        "temperature_c": field.temperature_c,
        "nitrogen_kg_ha": field.nitrogen_kg_ha,
        "phosphorus_kg_ha": field.phosphorus_kg_ha,
        "potassium_kg_ha": field.potassium_kg_ha,
        "irrigation_liters": field.irrigation_liters,
        "fertilizer_kg": field.fertilizer_kg,
        "energy_kwh": field.energy_kwh,
    }

    # apply scenario overrides
    for key in baseline:
        if key in scenario:
            baseline[key] = scenario[key]

    # Create one-row DataFrame
    candidate = pd.DataFrame([baseline])

    soil_moisture = models["soil_moisture_model"].predict(candidate)[0]
    crop_health = models["crop_health_model"].predict(candidate)[0]
    yield_tons = models["yield_model"].predict(candidate)[0]

    # Bound values to physical ranges
    soil_moisture = float(np.clip(soil_moisture, 10, 95))
    crop_health = float(np.clip(crop_health, 0, 100))
    yield_tons = float(np.clip(yield_tons, 0, 20))

    # Economic impact
    msp = MSP_PRICE.get(baseline["crop"], 2500)
    income_rs = yield_tons * baseline["area_ha"] * msp

    # Compare with MSP default baseline
    default_income = field.area_ha * field.msp_price_rs_per_ton * max(0.8, yield_tons * 0.7)

    return {
        "scenario": baseline,
        "predicted_soil_moisture_pct": round(soil_moisture, 2),
        "predicted_crop_health_score": round(crop_health, 2),
        "predicted_yield_tons": round(yield_tons, 2),
        "predicted_income_rs": round(income_rs, 2),
        "default_msp_income_rs": round(default_income, 2),
        "improvement_vs_default_pct": round(((income_rs - default_income) / default_income) * 100, 2) if default_income else 0.0,
    }


# ============================================================
# 6) Historical trend tracking
# ============================================================

def historical_trends(history_df: pd.DataFrame) -> Dict[str, Any]:
    """
    history_df must contain:
    date, crop, soil_moisture_pct, crop_health_score, yield_tons, irrigation_liters, fertilizer_kg, energy_kwh
    """
    history_df = history_df.copy()
    history_df["date"] = pd.to_datetime(history_df["date"])

    # Monthly or daily trend summary
    trend = (
        history_df.groupby(history_df["date"].dt.to_period("M"))
        .agg(
            avg_soil_moisture=("soil_moisture_pct", "mean"),
            avg_crop_health=("crop_health_score", "mean"),
            avg_yield=("yield_tons", "mean"),
            total_irrigation=("irrigation_liters", "sum"),
            total_fertilizer=("fertilizer_kg", "sum"),
            total_energy=("energy_kwh", "sum"),
        )
        .reset_index()
    )

    trend["date"] = trend["date"].astype(str)

    return {
        "monthly_trends": trend.to_dict(orient="records"),
    }


# ============================================================
# 7) Demo
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FARM DIGITAL TWIN - SCIKIT-LEARN DEMO")
    print("=" * 70)
    
    # Create synthetic farm data
    print("\n[1] Generating synthetic farm data (500 fields)...")
    df = generate_farm_data(500)
    print(f"✓ Generated {len(df)} records")
    print("\nSample data:")
    print(df[["field_id", "crop", "soil_type", "yield_tons", "income_rs"]].head())

    # Train models
    print("\n[2] Training ML models (soil moisture, crop health, yield)...")
    models = train_models(df)
    print("✓ Models trained successfully")

    # Evaluate models
    print("\n[3] Evaluating model performance...")
    X = df[
        [
            "crop",
            "soil_type",
            "area_ha",
            "rainfall_mm",
            "temperature_c",
            "nitrogen_kg_ha",
            "phosphorus_kg_ha",
            "potassium_kg_ha",
            "irrigation_liters",
            "fertilizer_kg",
            "energy_kwh",
        ]
    ]

    y_moisture = df["soil_moisture_pct"]
    y_health = df["crop_health_score"]
    y_yield = df["yield_tons"]

    moisture_r2 = r2_score(y_moisture, models["soil_moisture_model"].predict(X))
    health_r2 = r2_score(y_health, models["crop_health_model"].predict(X))
    yield_r2 = r2_score(y_yield, models["yield_model"].predict(X))

    print(f"Soil moisture model R² score: {moisture_r2:.3f}")
    print(f"Crop health model R² score: {health_r2:.3f}")
    print(f"Yield model R² score: {yield_r2:.3f}")

    # Dashboard
    print("\n[4] Farm Health Dashboard")
    print("-" * 70)
    dashboard = farm_health_dashboard(df)
    summary = dashboard["summary"]
    print(f"Total fields monitored: {summary['fields']}")
    print(f"Average soil moisture: {summary['average_soil_moisture_pct']}%")
    print(f"Average crop health score: {summary['average_crop_health_score']}/100")
    print(f"Average yield: {summary['average_yield_tons']} tons")
    print(f"Total water used: {summary['total_water_used_liters']:,.0f} liters")
    print(f"Total fertilizer: {summary['total_fertilizer_kg']:,.0f} kg")
    print(f"Total energy: {summary['total_energy_kwh']:,.0f} kWh")
    print(f"Average income per field: ₹{summary['average_income_rs']:,.0f}")
    
    print("\nCrop-wise summary:")
    for crop_data in dashboard["crop_summary"]:
        print(f"  {crop_data['crop'].upper()}: Health={crop_data['avg_health']:.1f}, Yield={crop_data['avg_yield']:.2f}t, Income=₹{crop_data['avg_income']:,.0f}")

    # Example field profile
    print("\n[5] Farm Digital Profile (Example Field)")
    print("-" * 70)
    sample_field = FarmField(
        field_id="F-001",
        crop="rice",
        soil_type="loam",
        area_ha=2.5,
        rainfall_mm=700,
        temperature_c=30,
        nitrogen_kg_ha=120,
        phosphorus_kg_ha=55,
        potassium_kg_ha=85,
        irrigation_liters=6500,
        fertilizer_kg=150,
        energy_kwh=220,
        msp_price_rs_per_ton=MSP_PRICE["rice"],
    )

    profile = get_farm_profile(sample_field)
    for key, value in profile.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # What-if scenario 1: Reduce irrigation
    print("\n[6] What-If Scenario Analysis")
    print("-" * 70)
    print("BASELINE (Current practice):")
    print(f"  Irrigation: {sample_field.irrigation_liters:.0f} liters")
    print(f"  Fertilizer: {sample_field.fertilizer_kg:.0f} kg")
    
    baseline_result = simulate_scenario(sample_field, {}, models)
    print(f"  → Predicted yield: {baseline_result['predicted_yield_tons']:.2f} tons")
    print(f"  → Predicted income: ₹{baseline_result['predicted_income_rs']:,.0f}")
    print(f"  → Crop health: {baseline_result['predicted_crop_health_score']:.1f}/100")

    print("\nSCENARIO 1: Reduce water usage by 20% (water conservation)")
    scenario1 = {
        "irrigation_liters": 5200,  # 20% reduction
        "fertilizer_kg": 140,
    }
    result1 = simulate_scenario(sample_field, scenario1, models)
    print(f"  Irrigation: 5200 liters (-20%)")
    print(f"  Fertilizer: 140 kg")
    print(f"  → Predicted yield: {result1['predicted_yield_tons']:.2f} tons")
    print(f"  → Predicted income: ₹{result1['predicted_income_rs']:,.0f}")
    print(f"  → Income change: {((result1['predicted_income_rs'] - baseline_result['predicted_income_rs']) / baseline_result['predicted_income_rs'] * 100):.1f}%")
    print(f"  → Crop health: {result1['predicted_crop_health_score']:.1f}/100")

    print("\nSCENARIO 2: Switch to wheat (crop rotation)")
    scenario2 = {
        "crop": "wheat",
        "irrigation_liters": 4500,
        "fertilizer_kg": 120,
        "nitrogen_kg_ha": 100,
    }
    sample_wheat = FarmField(
        field_id="F-001",
        crop="wheat",
        soil_type="loam",
        area_ha=2.5,
        rainfall_mm=700,
        temperature_c=30,
        nitrogen_kg_ha=100,
        phosphorus_kg_ha=50,
        potassium_kg_ha=75,
        irrigation_liters=4500,
        fertilizer_kg=120,
        energy_kwh=180,
        msp_price_rs_per_ton=MSP_PRICE["wheat"],
    )
    result2 = simulate_scenario(sample_wheat, {}, models)
    print(f"  Crop: wheat (rotated from rice)")
    print(f"  Irrigation: 4500 liters (-31%)")
    print(f"  Fertilizer: 120 kg (-20%)")
    print(f"  → Predicted yield: {result2['predicted_yield_tons']:.2f} tons")
    print(f"  → Predicted income: ₹{result2['predicted_income_rs']:,.0f}")
    print(f"  → Income change: {((result2['predicted_income_rs'] - baseline_result['predicted_income_rs']) / baseline_result['predicted_income_rs'] * 100):.1f}%")
    print(f"  → Crop health: {result2['predicted_crop_health_score']:.1f}/100")

    print("\nSCENARIO 3: Optimal fertilizer & reduced irrigation")
    scenario3 = {
        "irrigation_liters": 5500,
        "fertilizer_kg": 130,
        "nitrogen_kg_ha": 125,
        "phosphorus_kg_ha": 60,
        "potassium_kg_ha": 80,
    }
    result3 = simulate_scenario(sample_field, scenario3, models)
    print(f"  Irrigation: 5500 liters (-15%)")
    print(f"  Fertilizer: 130 kg (-13%)")
    print(f"  Nitrogen: 125 kg/ha (+4%)")
    print(f"  → Predicted yield: {result3['predicted_yield_tons']:.2f} tons")
    print(f"  → Predicted income: ₹{result3['predicted_income_rs']:,.0f}")
    print(f"  → Income change: {((result3['predicted_income_rs'] - baseline_result['predicted_income_rs']) / baseline_result['predicted_income_rs'] * 100):.1f}%")
    print(f"  → Crop health: {result3['predicted_crop_health_score']:.1f}/100")

    # Historical trends example
    print("\n[7] Historical Trends Analysis")
    print("-" * 70)
    history = df[["field_id", "crop", "soil_moisture_pct", "crop_health_score", "yield_tons", "irrigation_liters", "fertilizer_kg", "energy_kwh"]].copy()
    history["date"] = pd.date_range("2024-01-01", periods=len(history), freq="D")
    trend_result = historical_trends(history)
    print("Monthly trends (first 3 months):")
    for trend in trend_result["monthly_trends"][:3]:
        print(f"  {trend['date']}: Soil moisture={trend['avg_soil_moisture']:.1f}%, " 
              f"Health={trend['avg_crop_health']:.1f}, Yield={trend['avg_yield']:.2f}t")

    print("\n" + "=" * 70)
    print("SUMMARY: Digital Twin helps farmers:")
    print("  ✓ Understand current farm health in real-time")
    print("  ✓ Test irrigation/fertilizer changes before applying them")
    print("  ✓ Compare crop rotation options")
    print("  ✓ Optimize water and input usage while maintaining yield")
    print("  ✓ Track performance over time")
    print("=" * 70)
