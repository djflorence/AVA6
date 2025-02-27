"""
Backup utility for AVA6.

This script creates backups of the AVA6 codebase, conversation data, and vector database.
It can be run manually or scheduled as a cron job.
"""

import argparse
import datetime
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/backup.log"),
    ],
)
logger = logging.getLogger("backup")

# Default paths
DEFAULT_BACKUP_DIR = "backups"
DEFAULT_EXCLUDE_DIRS = [
    ".git",
    "venv",
    "env",
    "__pycache__",
    "logs",
    "backups",
    ".cursor",
]


def create_backup_directory(backup_dir: str) -> Path:
    """
    Create the backup directory if it doesn't exist.

    Args:
        backup_dir: Path to the backup directory

    Returns:
        Path object for the backup directory
    """
    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Backup directory: {backup_path.absolute()}")
    return backup_path


def create_timestamp() -> str:
    """
    Create a timestamp string for the backup filename.

    Returns:
        Timestamp string in the format YYYYMMDD_HHMMSS
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def backup_source_code(
    backup_path: Path, timestamp: str, exclude_dirs: List[str]
) -> Path:
    """
    Backup the source code.

    Args:
        backup_path: Path to the backup directory
        timestamp: Timestamp string for the backup filename
        exclude_dirs: List of directories to exclude from the backup

    Returns:
        Path to the backup file
    """
    logger.info("Backing up source code...")

    # Create a temporary directory for the source code
    temp_dir = backup_path / f"src_temp_{timestamp}"
    temp_dir.mkdir(exist_ok=True)

    # Copy the source code to the temporary directory
    for item in Path(".").iterdir():
        if item.name not in exclude_dirs and item.name != backup_path.name:
            if item.is_dir():
                shutil.copytree(
                    item,
                    temp_dir / item.name,
                    ignore=shutil.ignore_patterns(*exclude_dirs),
                )
            else:
                shutil.copy2(item, temp_dir / item.name)

    # Create a zip archive of the source code
    source_backup_file = backup_path / f"ava6_source_{timestamp}.zip"
    shutil.make_archive(str(source_backup_file).replace(".zip", ""), "zip", temp_dir)

    # Remove the temporary directory
    shutil.rmtree(temp_dir)

    logger.info(f"Source code backed up to {source_backup_file}")
    return source_backup_file


def backup_conversations(backup_path: Path, timestamp: str) -> Optional[Path]:
    """
    Backup the conversation data.

    Args:
        backup_path: Path to the backup directory
        timestamp: Timestamp string for the backup filename

    Returns:
        Path to the backup file, or None if no conversations found
    """
    logger.info("Backing up conversation data...")

    conversations_dir = Path("src/data/conversations")
    if not conversations_dir.exists():
        logger.warning("No conversation data found.")
        return None

    # Create a zip archive of the conversations
    conversations_backup_file = backup_path / f"ava6_conversations_{timestamp}.zip"
    shutil.make_archive(
        str(conversations_backup_file).replace(".zip", ""), "zip", conversations_dir
    )

    logger.info(f"Conversation data backed up to {conversations_backup_file}")
    return conversations_backup_file


def backup_vector_database(backup_path: Path, timestamp: str) -> Optional[Path]:
    """
    Backup the vector database.

    Args:
        backup_path: Path to the backup directory
        timestamp: Timestamp string for the backup filename

    Returns:
        Path to the backup file, or None if no vector database found
    """
    logger.info("Backing up vector database...")

    chroma_dir = Path("src/data/chroma")
    if not chroma_dir.exists():
        logger.warning("No vector database found.")
        return None

    # Create a zip archive of the vector database
    chroma_backup_file = backup_path / f"ava6_chroma_{timestamp}.zip"

    # We need to handle the SQLite database carefully
    # First, create a temporary directory
    temp_dir = backup_path / f"chroma_temp_{timestamp}"
    temp_dir.mkdir(exist_ok=True)

    # Copy the vector database to the temporary directory
    for item in chroma_dir.iterdir():
        if item.is_dir():
            shutil.copytree(item, temp_dir / item.name)
        else:
            # For SQLite files, we need to use a special approach
            if item.name.endswith(".sqlite3"):
                # Create a copy of the database
                shutil.copy2(item, temp_dir / item.name)
            else:
                shutil.copy2(item, temp_dir / item.name)

    # Create a zip archive of the vector database
    shutil.make_archive(str(chroma_backup_file).replace(".zip", ""), "zip", temp_dir)

    # Remove the temporary directory
    shutil.rmtree(temp_dir)

    logger.info(f"Vector database backed up to {chroma_backup_file}")
    return chroma_backup_file


def create_full_backup(
    backup_path: Path,
    timestamp: str,
    source_backup: Path,
    conversations_backup: Optional[Path],
    chroma_backup: Optional[Path],
) -> Path:
    """
    Create a full backup archive containing all individual backups.

    Args:
        backup_path: Path to the backup directory
        timestamp: Timestamp string for the backup filename
        source_backup: Path to the source code backup
        conversations_backup: Path to the conversations backup
        chroma_backup: Path to the vector database backup

    Returns:
        Path to the full backup file
    """
    logger.info("Creating full backup archive...")

    # Create a temporary directory for the full backup
    temp_dir = backup_path / f"full_temp_{timestamp}"
    temp_dir.mkdir(exist_ok=True)

    # Copy all individual backups to the temporary directory
    shutil.copy2(source_backup, temp_dir / source_backup.name)

    if conversations_backup:
        shutil.copy2(conversations_backup, temp_dir / conversations_backup.name)

    if chroma_backup:
        shutil.copy2(chroma_backup, temp_dir / chroma_backup.name)

    # Create a zip archive of all backups
    full_backup_file = backup_path / f"ava6_full_backup_{timestamp}.zip"
    shutil.make_archive(str(full_backup_file).replace(".zip", ""), "zip", temp_dir)

    # Remove the temporary directory
    shutil.rmtree(temp_dir)

    logger.info(f"Full backup created at {full_backup_file}")
    return full_backup_file


def cleanup_old_backups(backup_path: Path, keep_days: int) -> None:
    """
    Clean up old backups.

    Args:
        backup_path: Path to the backup directory
        keep_days: Number of days to keep backups
    """
    if keep_days <= 0:
        logger.info("Skipping cleanup of old backups.")
        return

    logger.info(f"Cleaning up backups older than {keep_days} days...")

    # Calculate the cutoff time
    cutoff_time = time.time() - (keep_days * 24 * 60 * 60)

    # Find and remove old backups
    for item in backup_path.iterdir():
        if item.is_file() and item.suffix == ".zip":
            if item.stat().st_mtime < cutoff_time:
                logger.info(f"Removing old backup: {item}")
                item.unlink()


def upload_to_cloud(backup_file: Path, cloud_provider: str) -> bool:
    """
    Upload the backup to a cloud storage service.

    Args:
        backup_file: Path to the backup file
        cloud_provider: Name of the cloud provider

    Returns:
        True if the upload was successful, False otherwise
    """
    logger.info(f"Uploading backup to {cloud_provider}...")

    # This is a placeholder for cloud upload functionality
    # Implement the specific cloud provider's API here

    if cloud_provider == "aws":
        # Example AWS S3 upload
        try:
            logger.info("AWS S3 upload not implemented yet.")
            return False
        except Exception as e:
            logger.error(f"Failed to upload to AWS S3: {str(e)}")
            return False
    elif cloud_provider == "google":
        # Example Google Cloud Storage upload
        try:
            logger.info("Google Cloud Storage upload not implemented yet.")
            return False
        except Exception as e:
            logger.error(f"Failed to upload to Google Cloud Storage: {str(e)}")
            return False
    elif cloud_provider == "azure":
        # Example Azure Blob Storage upload
        try:
            logger.info("Azure Blob Storage upload not implemented yet.")
            return False
        except Exception as e:
            logger.error(f"Failed to upload to Azure Blob Storage: {str(e)}")
            return False
    else:
        logger.error(f"Unsupported cloud provider: {cloud_provider}")
        return False

    return True


def main():
    """Main function to run the backup process."""
    parser = argparse.ArgumentParser(description="Backup utility for AVA6")
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=DEFAULT_BACKUP_DIR,
        help=f"Backup directory (default: {DEFAULT_BACKUP_DIR})",
    )
    parser.add_argument(
        "--exclude-dirs",
        type=str,
        nargs="+",
        default=DEFAULT_EXCLUDE_DIRS,
        help=f"Directories to exclude from the backup (default: {' '.join(DEFAULT_EXCLUDE_DIRS)})",
    )
    parser.add_argument(
        "--keep-days",
        type=int,
        default=30,
        help="Number of days to keep backups (default: 30, 0 to disable cleanup)",
    )
    parser.add_argument(
        "--cloud-upload",
        type=str,
        choices=["aws", "google", "azure"],
        help="Upload backup to cloud storage (aws, google, azure)",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Keep individual backup files (source, conversations, chroma)",
    )

    args = parser.parse_args()

    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    # Create backup directory
    backup_path = create_backup_directory(args.backup_dir)

    # Create timestamp
    timestamp = create_timestamp()

    # Backup source code
    source_backup = backup_source_code(backup_path, timestamp, args.exclude_dirs)

    # Backup conversations
    conversations_backup = backup_conversations(backup_path, timestamp)

    # Backup vector database
    chroma_backup = backup_vector_database(backup_path, timestamp)

    # Create full backup
    full_backup = create_full_backup(
        backup_path, timestamp, source_backup, conversations_backup, chroma_backup
    )

    # Remove individual backups if not needed
    if not args.individual:
        source_backup.unlink()
        if conversations_backup:
            conversations_backup.unlink()
        if chroma_backup:
            chroma_backup.unlink()

    # Upload to cloud if requested
    if args.cloud_upload:
        upload_to_cloud(full_backup, args.cloud_upload)

    # Clean up old backups
    cleanup_old_backups(backup_path, args.keep_days)

    logger.info("Backup completed successfully.")


if __name__ == "__main__":
    main()
