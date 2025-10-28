"""Dataset downloaders for various sources."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List
from urllib.parse import urlparse

import requests
from eliot import start_action
from fsspec import open as fsspec_open
from fsspec.implementations.http import HTTPFileSystem

from glucose_neuralforecast.glucoseml.registry import DatasetConfig


def _load_env_from_dotenv() -> None:
    """Load PHYSIONET credentials from a .env file at project root if present.

    We avoid adding a new dependency and parse a simple KEY=VALUE format.
    Only sets variables that are not already present in the environment.
    """
    try:
        # Resolve project base folder without importing heavy modules here
        from glucose_neuralforecast.utils import resolve_base_folder
        base = resolve_base_folder()
        dotenv_path = base / '.env'
        if not dotenv_path.exists():
            return
        for line in dotenv_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and (key not in os.environ):
                os.environ[key] = value
    except Exception:
        # Do not fail download if .env parsing fails; proceed with existing env
        pass


def _physionet_files_base_url(content_url: str) -> str:
    """Transform PhysioNet content URL to files URL for authenticated access.

    Example:
      https://physionet.org/content/cgmacros/1.0.0/ ->
      https://physionet.org/files/cgmacros/1.0.0/
    """
    return content_url.replace('/content/', '/files/')


def download_physionet(dataset_config: DatasetConfig, output_dir: Path) -> None:
    """
    Download dataset from PhysioNet using wget/fsspec.
    
    Args:
        dataset_config: Dataset configuration from registry
        output_dir: Directory to save downloaded files
    """
    with start_action(
        action_type="download_physionet",
        dataset=dataset_config.name,
        source=dataset_config.source,
        output_dir=str(output_dir)
    ) as action:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # PhysioNet URLs typically point to a directory listing
        # For BIG_IDEAS: https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/
        # Files are: Dexcom_001.csv, Dexcom_002.csv, ... Dexcom_016.csv
        
        # Load credentials from environment or .env
        _load_env_from_dotenv()
        username = os.getenv('PHYSIONET_USERNAME')
        password = os.getenv('PHYSIONET_PASSWORD')

        if not username or not password:
            action.log(message_type="missing_credentials")
            raise RuntimeError(
                "PhysioNet credentials not found. Set PHYSIONET_USERNAME and PHYSIONET_PASSWORD (e.g., in .env)."
            )

        if 'BIG_IDEAS' in dataset_config.name or 'big-ideas' in dataset_config.source.lower():
            # Download specific files based on subject range
            subject_range = dataset_config.preprocessing.subject_range
            if subject_range:
                start_subj, end_subj = subject_range
                base_url = _physionet_files_base_url(dataset_config.source).rstrip('/')
                for subj_num in range(start_subj, end_subj + 1):
                    file_name = f"Dexcom_{subj_num:03d}.csv"
                    # Files are organized under /{subject}/Dexcom_{subject}.csv
                    file_url = f"{base_url}/{subj_num:03d}/{file_name}"
                    output_file = output_dir / file_name
                    
                    if output_file.exists():
                        action.log(message_type="file_exists", file=file_name)
                        continue
                    
                    action.log(message_type="downloading_file", file=file_name, url=file_url, user=username)
                    
                    try:
                        # Use HTTP basic auth against the /files/ endpoint
                        with requests.get(file_url, auth=(username, password), stream=True, allow_redirects=True) as resp:
                            resp.raise_for_status()
                            with open(output_file, 'wb') as f_out:
                                for chunk in resp.iter_content(chunk_size=8192):
                                    if chunk:
                                        f_out.write(chunk)
                        
                        action.log(message_type="file_downloaded", file=file_name, size=output_file.stat().st_size)
                    except Exception as e:
                        action.log(message_type="download_error", file=file_name, error=str(e))
                        raise
        
        elif 'CGMacros' in dataset_config.name or 'cgmacros' in dataset_config.source.lower():
            # CGMacros: Download zip file and extract (dateshifted public release)
            zip_url = f"{_physionet_files_base_url(dataset_config.source).rstrip('/')}/CGMacros_dateshifted365.zip"
            zip_file = output_dir / "CGMacros_dateshifted365.zip"
            extracted_dir = output_dir / "CGMacros"
            
            # Check if already extracted
            if extracted_dir.exists() and any(extracted_dir.iterdir()):
                action.log(message_type="already_extracted", path=str(extracted_dir))
            else:
                if not zip_file.exists():
                    action.log(message_type="downloading_zip", url=zip_url, user=username)
                    
                    with requests.get(zip_url, auth=(username, password), stream=True, allow_redirects=True) as resp:
                        resp.raise_for_status()
                        with open(zip_file, 'wb') as f_out:
                            for chunk in resp.iter_content(chunk_size=8192):
                                if chunk:
                                    f_out.write(chunk)
                    
                    action.log(message_type="zip_downloaded", size=zip_file.stat().st_size)
                else:
                    action.log(message_type="zip_exists", path=str(zip_file))
                
                # Extract zip
                action.log(message_type="extracting_zip")
                import zipfile
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                
                action.log(message_type="extraction_complete")
        
        else:
            raise NotImplementedError(f"PhysioNet download for {dataset_config.name} not implemented")
        
        action.log(message_type="download_complete", dataset=dataset_config.name)


def download_figshare(dataset_config: DatasetConfig, output_dir: Path) -> None:
    """
    Download dataset from Figshare using API.
    
    Args:
        dataset_config: Dataset configuration from registry
        output_dir: Directory to save downloaded files
    """
    with start_action(
        action_type="download_figshare",
        dataset=dataset_config.name,
        source=dataset_config.source,
        output_dir=str(output_dir)
    ) as action:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract collection ID from URL
        # Example: https://figshare.com/collections/Shanghai_Type_1_Diabetes_Mellitus_Dataset/6310860
        collection_id = dataset_config.source.rstrip('/').split('/')[-1]
        
        action.log(message_type="fetching_collection", collection_id=collection_id)
        
        # Get collection metadata
        collection_url = f"https://api.figshare.com/v2/collections/{collection_id}/articles"
        response = requests.get(collection_url)
        response.raise_for_status()
        articles = response.json()
        
        action.log(message_type="collection_fetched", num_articles=len(articles))
        
        # Download each article's files
        for article in articles:
            article_id = article['id']
            article_title = article['title']
            
            action.log(message_type="fetching_article", article_id=article_id, title=article_title)
            
            # Get article details with file list
            article_url = f"https://api.figshare.com/v2/articles/{article_id}"
            response = requests.get(article_url)
            response.raise_for_status()
            article_data = response.json()
            
            files = article_data.get('files', [])
            action.log(message_type="article_files", num_files=len(files))
            
            for file_info in files:
                file_name = file_info['name']
                file_url = file_info['download_url']
                file_size = file_info['size']
                
                output_file = output_dir / file_name
                
                if output_file.exists():
                    action.log(message_type="file_exists", file=file_name)
                    continue
                
                action.log(message_type="downloading_file", file=file_name, size=file_size)
                
                try:
                    response = requests.get(file_url, stream=True)
                    response.raise_for_status()
                    
                    with open(output_file, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    action.log(message_type="file_downloaded", file=file_name)
                except Exception as e:
                    action.log(message_type="download_error", file=file_name, error=str(e))
                    raise
        
        action.log(message_type="download_complete", dataset=dataset_config.name)


def download_github(dataset_config: DatasetConfig, output_dir: Path) -> None:
    """
    Download dataset from GitHub using git clone.
    
    Args:
        dataset_config: Dataset configuration from registry
        output_dir: Directory to save downloaded files
    """
    with start_action(
        action_type="download_github",
        dataset=dataset_config.name,
        source=dataset_config.source,
        output_dir=str(output_dir)
    ) as action:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clone the repository
        repo_url = dataset_config.source
        
        # Check if already cloned
        repo_name = repo_url.rstrip('/').split('/')[-1]
        repo_dir = output_dir / repo_name
        
        if repo_dir.exists():
            action.log(message_type="repo_exists", path=str(repo_dir))
            # Optionally update: git pull
            try:
                subprocess.run(
                    ['git', 'pull'],
                    cwd=repo_dir,
                    check=True,
                    capture_output=True,
                    text=True
                )
                action.log(message_type="repo_updated")
            except subprocess.CalledProcessError as e:
                action.log(message_type="update_error", error=str(e))
        else:
            action.log(message_type="cloning_repo", url=repo_url)
            
            try:
                subprocess.run(
                    ['git', 'clone', repo_url, str(repo_dir)],
                    check=True,
                    capture_output=True,
                    text=True
                )
                action.log(message_type="clone_complete")
            except subprocess.CalledProcessError as e:
                action.log(message_type="clone_error", error=str(e))
                raise
        
        action.log(message_type="download_complete", dataset=dataset_config.name)


def download_dataset(
    dataset_name: str,
    output_dir: Path,
    dataset_config: DatasetConfig
) -> None:
    """
    Download a dataset based on its configuration.
    
    Args:
        dataset_name: Name of the dataset
        output_dir: Directory to save downloaded files
        dataset_config: Dataset configuration from registry
    """
    with start_action(
        action_type="download_dataset",
        dataset=dataset_name,
        output_dir=str(output_dir)
    ) as action:
        source = dataset_config.source
        
        # Determine download method based on source
        if 'physionet.org' in source:
            download_physionet(dataset_config, output_dir)
        elif 'figshare.com' in source:
            download_figshare(dataset_config, output_dir)
        elif 'github.com' in source:
            download_github(dataset_config, output_dir)
        else:
            raise ValueError(f"Unknown source type for dataset {dataset_name}: {source}")
        
        action.log(message_type="dataset_downloaded", dataset=dataset_name)


def download_all_datasets(
    datasets: List[str],
    base_output_dir: Path,
    registry_config: Optional[object] = None
) -> None:
    """
    Download multiple datasets.
    
    Args:
        datasets: List of dataset names to download
        base_output_dir: Base directory for raw downloads
        registry_config: Optional registry configuration (loaded if None)
    """
    with start_action(
        action_type="download_all_datasets",
        datasets=datasets,
        output_dir=str(base_output_dir)
    ) as action:
        if registry_config is None:
            from glucose_neuralforecast.glucoseml.registry import load_registry
            registry_config = load_registry()
        
        for dataset_name in datasets:
            if dataset_name not in registry_config.datasets:
                action.log(message_type="dataset_not_found", dataset=dataset_name)
                continue
            
            dataset_config = registry_config.datasets[dataset_name]
            output_dir = base_output_dir / dataset_name
            
            try:
                download_dataset(dataset_name, output_dir, dataset_config)
            except Exception as e:
                action.log(message_type="download_failed", dataset=dataset_name, error=str(e))
                raise
        
        action.log(message_type="all_downloads_complete")

