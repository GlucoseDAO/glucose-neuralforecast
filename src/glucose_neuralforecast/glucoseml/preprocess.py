"""Dataset preprocessing functions to produce minimal schema."""

import glob
from pathlib import Path
from typing import Optional, List

import polars as pl
from eliot import start_action

from glucose_neuralforecast.glucoseml.registry import DatasetConfig


def preprocess_big_ideas(
    input_dir: Path,
    output_dir: Path,
    dataset_config: DatasetConfig
) -> None:
    """
    Preprocess BIG IDEAs dataset to minimal schema.
    
    Args:
        input_dir: Directory containing raw CSV files
        output_dir: Directory to save preprocessed CSV files
        dataset_config: Dataset configuration
    """
    with start_action(
        action_type="preprocess_big_ideas",
        input_dir=str(input_dir),
        output_dir=str(output_dir)
    ) as action:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        preprocessing = dataset_config.preprocessing
        
        # Find all Dexcom_*.csv files
        csv_files = list(input_dir.glob("Dexcom_*.csv"))
        action.log(message_type="files_found", count=len(csv_files))
        
        for csv_file in csv_files:
            # Extract subject ID from filename (e.g., Dexcom_001.csv -> 001)
            subject_id = csv_file.stem.split('_')[1]
            
            action.log(message_type="processing_subject", subject_id=subject_id, file=csv_file.name)
            
            # Read CSV
            df = pl.read_csv(csv_file)
            
            # Filter EGV rows
            if preprocessing.filter_event_type:
                df = df.filter(pl.col("Event Type") == preprocessing.filter_event_type)
            
            # Select and rename columns
            select_exprs = [
                pl.col(preprocessing.timestamp_column).alias("timestamp"),
                pl.col(preprocessing.glucose_column).alias("BGvalue")
            ]
            # Optionally include exogenous columns if configured and present
            if getattr(preprocessing, "exogenous_columns", None):
                for ex_col in preprocessing.exogenous_columns or []:
                    if ex_col in df.columns:
                        select_exprs.append(pl.col(ex_col))
            df_minimal = df.select(select_exprs)
            
            # Save to output
            output_file = output_dir / f"{subject_id}.csv"
            df_minimal.write_csv(output_file)
            
            action.log(
                message_type="subject_processed",
                subject_id=subject_id,
                rows=len(df_minimal),
                output=str(output_file)
            )
        
        action.log(message_type="preprocessing_complete", subjects_processed=len(csv_files))


def preprocess_shanghai(
    input_dir: Path,
    output_dir: Path,
    dataset_config: DatasetConfig,
    dataset_name: str
) -> None:
    """
    Preprocess Shanghai T1DM or T2DM dataset to minimal schema.
    
    Args:
        input_dir: Directory containing raw Excel files
        output_dir: Directory to save preprocessed CSV files
        dataset_config: Dataset configuration
        dataset_name: Name of dataset (for logging)
    """
    with start_action(
        action_type="preprocess_shanghai",
        dataset=dataset_name,
        input_dir=str(input_dir),
        output_dir=str(output_dir)
    ) as action:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        preprocessing = dataset_config.preprocessing
        
        # Find all Excel files
        xlsx_files = list(input_dir.glob("*.xlsx")) + list(input_dir.glob("*.xls"))
        action.log(message_type="files_found", count=len(xlsx_files))
        
        # Group files by subject (prefix before first underscore)
        subject_files: dict[str, List[Path]] = {}
        for file in xlsx_files:
            subject_id = file.stem.split('_')[0]
            if subject_id not in subject_files:
                subject_files[subject_id] = []
            subject_files[subject_id].append(file)
        
        action.log(message_type="subjects_found", count=len(subject_files))
        
        for subject_id, files in subject_files.items():
            action.log(message_type="processing_subject", subject_id=subject_id, num_files=len(files))
            
            # Sort files for consistent ordering
            files.sort()
            
            dfs = []
            for file in files:
                try:
                    # Read Excel file
                    df = pl.read_excel(file)
                    
                    # Try primary glucose column, then alt column
                    glucose_col = preprocessing.glucose_column
                    if glucose_col not in df.columns and preprocessing.glucose_column_alt:
                        glucose_col = preprocessing.glucose_column_alt
                    
                    if glucose_col not in df.columns:
                        action.log(
                            message_type="glucose_column_not_found",
                            file=file.name,
                            available_columns=df.columns
                        )
                        continue
                    
                    # Select and rename columns
                    select_exprs = [
                        pl.col(preprocessing.timestamp_column).alias("timestamp"),
                        pl.col(glucose_col).alias("BGvalue")
                    ]
                    if getattr(preprocessing, "exogenous_columns", None):
                        for ex_col in preprocessing.exogenous_columns or []:
                            if ex_col in df.columns:
                                select_exprs.append(pl.col(ex_col))
                    df_minimal = df.select(select_exprs)
                    
                    dfs.append(df_minimal)
                    
                except Exception as e:
                    action.log(message_type="file_error", file=file.name, error=str(e))
                    continue
            
            if dfs:
                # Collect all unique columns across files for this subject
                all_cols = set()
                for df in dfs:
                    all_cols.update(df.columns)
                all_cols.discard("timestamp")
                all_cols.discard("BGvalue")
                covariate_cols = sorted(all_cols)
                
                # Align schemas: ensure all dfs have the same columns
                aligned_dfs = []
                for df in dfs:
                    cast_exprs = [
                        pl.col("timestamp"),
                        pl.col("BGvalue").cast(pl.Float64, strict=False)
                    ]
                    for cov_col in covariate_cols:
                        if cov_col in df.columns:
                            cast_exprs.append(pl.col(cov_col).cast(pl.Float64, strict=False))
                        else:
                            cast_exprs.append(pl.lit(None).cast(pl.Float64).alias(cov_col))
                    aligned_dfs.append(df.select(cast_exprs))
                
                # Concatenate all files for this subject
                if len(aligned_dfs) > 1:
                    df_combined = pl.concat(aligned_dfs)
                else:
                    df_combined = aligned_dfs[0]
                
                # Save to output
                output_file = output_dir / f"{subject_id}.csv"
                df_combined.write_csv(output_file)
                
                action.log(
                    message_type="subject_processed",
                    subject_id=subject_id,
                    rows=len(df_combined),
                    output=str(output_file)
                )
            else:
                action.log(message_type="subject_skipped", subject_id=subject_id, reason="no_valid_files")
        
        action.log(message_type="preprocessing_complete", subjects_processed=len(subject_files))


def preprocess_uchtt1dm(
    input_dir: Path,
    output_dir: Path,
    dataset_config: DatasetConfig
) -> None:
    """
    Preprocess UCHTT1DM dataset to minimal schema.
    
    This dataset has separate Excel files for each variable:
    - Glucose.xlsx (timestamp, glucose value)
    - Carbohidrates.xlsx (timestamp, carb value)
    - IGAR.xlsx (timestamp, IGAR value)
    - Steps.xlsx (timestamp, steps value)
    - Heart Rate.xlsx (timestamp, heart rate values)
    
    We join them all on timestamp.
    
    Args:
        input_dir: Directory containing raw data (GitHub repo)
        output_dir: Directory to save preprocessed CSV files
        dataset_config: Dataset configuration
    """
    with start_action(
        action_type="preprocess_uchtt1dm",
        input_dir=str(input_dir),
        output_dir=str(output_dir)
    ) as action:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        preprocessing = dataset_config.preprocessing
        
        # Find repository directory (UC_HT_T1DM)
        repo_dirs = [d for d in input_dir.iterdir() if d.is_dir() and 'UC_HT_T1DM' in d.name]
        if not repo_dirs:
            raise FileNotFoundError(f"UC_HT_T1DM repository not found in {input_dir}")
        
        repo_dir = repo_dirs[0]
        action.log(message_type="repo_found", path=str(repo_dir))
        
        # Find all subject folders containing Glucose.xlsx
        glucose_files = list(repo_dir.glob("*/Glucose.xlsx"))
        action.log(message_type="glucose_files_found", count=len(glucose_files))
        
        for glucose_file in glucose_files:
            # Subject ID is the parent folder name
            subject_id = glucose_file.parent.name
            subject_folder = glucose_file.parent
            
            action.log(message_type="processing_subject", subject_id=subject_id, file=str(glucose_file))
            
            try:
                # Read glucose file
                df_glucose = pl.read_excel(glucose_file)
                df_glucose = df_glucose.rename({
                    preprocessing.timestamp_column: "timestamp",
                    preprocessing.glucose_column: "BGvalue"
                })
                
                # Keep only timestamp and BGvalue, plus Value_interp if present
                base_cols = ["timestamp", "BGvalue"]
                if "Value_interp (mg/dl)" in df_glucose.columns:
                    base_cols.append("Value_interp (mg/dl)")
                df_glucose = df_glucose.select(base_cols)
                
                # Join with covariate files if they exist
                covariate_files = {
                    "Carbohidrates.xlsx": "Carbs (g)",
                    "IGAR.xlsx": "IGAR (g)",
                    "Steps.xlsx": "Steps",
                    "Heart Rate.xlsx": "Heart Rate (bpm)"
                }
                
                for file_name, col_rename in covariate_files.items():
                    covariate_path = subject_folder / file_name
                    if covariate_path.exists():
                        try:
                            df_cov = pl.read_excel(covariate_path)
                            # Rename timestamp column
                            df_cov = df_cov.rename({"__UNNAMED__0": "timestamp"})
                            
                            # For Heart Rate, prefer non-interpolated Value
                            if file_name == "Heart Rate.xlsx":
                                if "Value (bpm)" in df_cov.columns:
                                    df_cov = df_cov.select(["timestamp", pl.col("Value (bpm)").alias(col_rename)])
                                elif "Value_interp (bpm)" in df_cov.columns:
                                    df_cov = df_cov.select(["timestamp", pl.col("Value_interp (bpm)").alias(col_rename)])
                            else:
                                # For other files, rename Value column
                                value_col = [c for c in df_cov.columns if c.startswith("Value")]
                                if value_col:
                                    df_cov = df_cov.select(["timestamp", pl.col(value_col[0]).alias(col_rename)])
                            
                            # Join with main dataframe on timestamp
                            df_glucose = df_glucose.join(df_cov, on="timestamp", how="left")
                            action.log(message_type="joined_covariate", file=file_name, column=col_rename)
                        except Exception as e:
                            action.log(message_type="covariate_read_error", file=file_name, error=str(e))
                
                # Drop null BGvalue rows
                df_minimal = df_glucose.drop_nulls(subset=["BGvalue"])
                
                # Save to output
                output_file = output_dir / f"{subject_id}.csv"
                df_minimal.write_csv(output_file)
                
                action.log(
                    message_type="subject_processed",
                    subject_id=subject_id,
                    rows=len(df_minimal),
                    columns=df_minimal.columns,
                    output=str(output_file)
                )
                
            except Exception as e:
                action.log(message_type="processing_error", subject_id=subject_id, error=str(e))
                continue
        
        action.log(message_type="preprocessing_complete", subjects_processed=len(glucose_files))


def preprocess_cgmacros(
    input_dir: Path,
    output_dir: Path,
    dataset_config: DatasetConfig
) -> None:
    """
    Preprocess CGMacros dataset to minimal schema.
    
    Args:
        input_dir: Directory containing extracted CGMacros data
        output_dir: Directory to save preprocessed CSV files
        dataset_config: Dataset configuration
    """
    with start_action(
        action_type="preprocess_cgmacros",
        input_dir=str(input_dir),
        output_dir=str(output_dir)
    ) as action:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        preprocessing = dataset_config.preprocessing
        
        # Find all subject folders; support both Subject_* and CGMacros-* layouts
        subject_folders = []
        for root_folder in input_dir.iterdir():
            if not root_folder.is_dir():
                continue
            subject_folders.extend([d for d in root_folder.glob("Subject_*") if d.is_dir()])
            subject_folders.extend([d for d in root_folder.glob("CGMacros-*") if d.is_dir()])
        
        if not subject_folders:
            subject_folders = list(input_dir.glob("Subject_*"))
        if not subject_folders:
            subject_folders = list(input_dir.glob("CGMacros-*") )
        
        action.log(message_type="subject_folders_found", count=len(subject_folders))
        
        for subject_folder in subject_folders:
            subject_id = subject_folder.name
            
            action.log(message_type="processing_subject", subject_id=subject_id)
            
            # Find CSV file in subject folder
            csv_files = list(subject_folder.glob("CGM_*.csv"))
            if not csv_files:
                csv_files = list(subject_folder.glob("CGMacros-*.csv"))
            if not csv_files:
                csv_files = list(subject_folder.glob("*.csv"))
            
            if not csv_files:
                action.log(message_type="no_csv_found", subject_id=subject_id)
                continue
            
            csv_file = csv_files[0]
            
            try:
                # Read CSV
                df = pl.read_csv(csv_file)
                
                # Select glucose column based on sensor preference
                glucose_col = None
                for pref_col in preprocessing.sensor_preference:
                    if pref_col in df.columns:
                        glucose_col = pref_col
                        break
                
                if glucose_col is None:
                    action.log(
                        message_type="no_glucose_column",
                        subject_id=subject_id,
                        available_columns=df.columns
                    )
                    continue
                
                # Select and rename columns, including configured exogenous if available
                select_exprs = [
                    pl.col(preprocessing.timestamp_column).alias("timestamp"),
                    pl.col(glucose_col).alias("BGvalue")
                ]
                if getattr(preprocessing, "exogenous_columns", None):
                    for ex_col in preprocessing.exogenous_columns or []:
                        if ex_col in df.columns:
                            select_exprs.append(pl.col(ex_col))
                df_minimal = df.select(select_exprs)
                
                # Drop null BGvalue rows
                df_minimal = df_minimal.drop_nulls(subset=["BGvalue"])
                
                # Save to output
                output_file = output_dir / f"{subject_id}.csv"
                df_minimal.write_csv(output_file)
                
                action.log(
                    message_type="subject_processed",
                    subject_id=subject_id,
                    glucose_column=glucose_col,
                    rows=len(df_minimal),
                    output=str(output_file)
                )
                
            except Exception as e:
                action.log(message_type="processing_error", subject_id=subject_id, error=str(e))
                continue
        
        action.log(message_type="preprocessing_complete", subjects_processed=len(subject_folders))


def preprocess_dataset_to_parquet(
    dataset_name: str,
    input_dir: Path,
    output_parquet: Path,
    dataset_config: DatasetConfig,
    validation_config: Optional[object] = None
) -> dict:
    """
    Preprocess a dataset with optional inline validation and write directly to parquet.
    
    Args:
        dataset_name: Name of the dataset
        input_dir: Directory containing raw data
        output_parquet: Path to output parquet file
        dataset_config: Dataset configuration
        validation_config: Optional validation configuration for inline validation
    
    Returns:
        Dictionary with dataset statistics
    """
    with start_action(
        action_type="preprocess_dataset_to_parquet",
        dataset=dataset_name,
        input_dir=str(input_dir),
        output=str(output_parquet)
    ) as action:
        # Use temporary directory for intermediate CSVs
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_output = Path(temp_dir)
            
            # Run dataset-specific preprocessing to temp CSVs
            if dataset_name == "BIG_IDEAS":
                preprocess_big_ideas(input_dir, temp_output, dataset_config)
            elif dataset_name == "ShanghaiT1DM" or dataset_name == "ShanghaiT2DM":
                preprocess_shanghai(input_dir, temp_output, dataset_config, dataset_name)
            elif dataset_name == "UCHTT1DM":
                preprocess_uchtt1dm(input_dir, temp_output, dataset_config)
            elif dataset_name == "CGMacros":
                preprocess_cgmacros(input_dir, temp_output, dataset_config)
            else:
                raise NotImplementedError(f"Preprocessing for {dataset_name} not implemented")
            
            # Load and combine all subject CSVs
            csv_files = list(temp_output.glob("*.csv"))
            action.log(message_type="temp_csvs_created", count=len(csv_files))
            
            # First pass: collect all unique covariate column names
            all_columns = set()
            for csv_file in csv_files:
                df_temp = pl.read_csv(csv_file, infer_schema_length=10000, n_rows=1)
                all_columns.update(df_temp.columns)
            
            # Remove timestamp and BGvalue (these are required)
            all_columns.discard("timestamp")
            all_columns.discard("BGvalue")
            covariate_columns = sorted(all_columns)  # Sort for consistent ordering
            
            action.log(message_type="covariate_columns_detected", columns=covariate_columns)
            
            all_subject_dfs = []
            
            for csv_file in csv_files:
                df = pl.read_csv(csv_file, infer_schema_length=10000)
                if len(df) == 0:
                    continue
                
                # Standardize all covariate column types to Float64
                # This handles mixed-type issues (int vs float vs string) in covariate columns
                cast_exprs = [
                    pl.col("timestamp"),  # Keep timestamp as-is
                    pl.col("BGvalue").cast(pl.Float64, strict=False)
                ]
                
                # Add all covariate columns, filling missing ones with null
                for cov_col in covariate_columns:
                    if cov_col in df.columns:
                        cast_exprs.append(pl.col(cov_col).cast(pl.Float64, strict=False))
                    else:
                        cast_exprs.append(pl.lit(None).cast(pl.Float64).alias(cov_col))
                
                df = df.select(cast_exprs)
                
                # Add dataset metadata
                subject_id = csv_file.stem
                df = df.with_columns([
                    pl.lit(dataset_name).alias("dataset"),
                    pl.lit(subject_id).alias("subject_id"),
                    pl.lit(f"{dataset_name}__{subject_id}").alias("unique_id"),
                    pl.col("timestamp").alias("ds"),
                    pl.col("BGvalue").alias("y"),
                    pl.lit(dataset_config.sensor_family).alias("sensor_family"),
                    pl.lit(dataset_config.sampling_minutes).alias("sampling_minutes"),
                    pl.lit(dataset_config.timezone).alias("timezone")
                ])
                
                # Convert timestamp if needed
                if df["ds"].dtype == pl.Utf8:
                    df = df.with_columns(pl.col("ds").str.to_datetime())
                
                # Pass through covariates
                non_cov_cols = {"timestamp", "BGvalue", "dataset", "subject_id", "unique_id", "ds", "y", "sensor_family", "sampling_minutes", "timezone"}
                passthrough_cols = [c for c in df.columns if c not in non_cov_cols]
                
                df = df.select([
                    "dataset", "subject_id", "unique_id", "ds", "y",
                    "sensor_family", "sampling_minutes", "timezone",
                    *passthrough_cols
                ])
                
                all_subject_dfs.append(df)
            
            if not all_subject_dfs:
                raise ValueError(f"No valid subjects for {dataset_name}")
            
            # Concatenate with vertical (schema should now be consistent due to pre-casting)
            combined = pl.concat(all_subject_dfs, how="vertical")
            combined = combined.sort(["unique_id", "ds"])
            
            output_parquet.parent.mkdir(parents=True, exist_ok=True)
            combined.write_parquet(output_parquet)
            
            # Collect stats
            non_cov_cols = {"dataset", "subject_id", "unique_id", "ds", "y", "sensor_family", "sampling_minutes", "timezone"}
            covariate_cols = [c for c in combined.columns if c not in non_cov_cols]
            
            stats = {
                "dataset": dataset_name,
                "total_glucose_values": len(combined),
                "subjects": combined["unique_id"].n_unique(),
                "date_range": f"{combined['ds'].min()} to {combined['ds'].max()}",
                "covariates": covariate_cols
            }
            
            action.log(message_type="parquet_written", output=str(output_parquet), stats=stats)
            
            return stats


def preprocess_all_datasets(
    datasets: List[str],
    base_input_dir: Path,
    base_output_dir: Path,
    registry_config: Optional[object] = None,
    apply_validation: bool = True
) -> dict:
    """
    Preprocess multiple datasets to parquet files with inline validation.
    
    Args:
        datasets: List of dataset names to preprocess
        base_input_dir: Base directory for raw data
        base_output_dir: Base directory for output parquet files
        registry_config: Optional registry configuration (loaded if None)
        apply_validation: Whether to apply validation inline (default: True)
    
    Returns:
        Dictionary with all dataset statistics
    """
    with start_action(
        action_type="preprocess_all_datasets",
        datasets=datasets,
        input_dir=str(base_input_dir),
        output_dir=str(base_output_dir)
    ) as action:
        if registry_config is None:
            from glucose_neuralforecast.glucoseml.registry import load_registry
            registry_config = load_registry()
        
        all_stats = {}
        
        for dataset_name in datasets:
            if dataset_name not in registry_config.datasets:
                action.log(message_type="dataset_not_found", dataset=dataset_name)
                continue
            
            dataset_config = registry_config.datasets[dataset_name]
            input_dir = base_input_dir / dataset_name
            output_parquet = base_output_dir / f"{dataset_name}.parquet"
            
            if not input_dir.exists():
                action.log(message_type="input_dir_not_found", dataset=dataset_name, path=str(input_dir))
                continue
            
            try:
                stats = preprocess_dataset_to_parquet(
                    dataset_name,
                    input_dir,
                    output_parquet,
                    dataset_config,
                    validation_config=registry_config.validation if apply_validation else None
                )
                all_stats[dataset_name] = stats
            except Exception as e:
                action.log(message_type="preprocessing_failed", dataset=dataset_name, error=str(e))
                raise
        
        # Write summary stats
        stats_dir = base_output_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        # Per-dataset CSV
        per_ds_rows = []
        for ds_name, ds_stats in all_stats.items():
            per_ds_rows.append({
                "dataset": ds_name,
                "total_glucose_values": ds_stats["total_glucose_values"],
                "subjects": ds_stats["subjects"],
                "date_range": ds_stats["date_range"],
                "covariates_str": "|".join(ds_stats.get("covariates", []))
            })
        
        if per_ds_rows:
            per_ds_df = pl.DataFrame(per_ds_rows)
            per_ds_csv = stats_dir / "per_dataset_counts.csv"
            per_ds_df.write_csv(per_ds_csv)
            action.log(message_type="stats_csv_saved", file=str(per_ds_csv))
        
        # Summary YAML
        import yaml
        summary = {
            "total_rows_sum": sum(s["total_glucose_values"] for s in all_stats.values()),
            "total_subjects_sum": sum(s["subjects"] for s in all_stats.values()),
            "datasets_included": len(all_stats),
            "by_dataset": {ds: {
                "rows": s["total_glucose_values"],
                "subjects": s["subjects"],
                "date_range": s["date_range"],
                "covariates": s.get("covariates", [])
            } for ds, s in all_stats.items()}
        }
        stats_yaml = stats_dir / "stats.yaml"
        stats_yaml.write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True))
        action.log(message_type="stats_yaml_saved", file=str(stats_yaml))
        
        action.log(message_type="all_preprocessing_complete", datasets=len(all_stats))
        
        return all_stats

