"""
Test script for the backup utility.

This script tests the backup utility by creating a backup and verifying that the backup files exist.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from src.utils.backup import (
    backup_source_code,
    backup_conversations,
    backup_vector_database,
    create_full_backup,
    create_backup_directory,
    create_timestamp,
    cleanup_old_backups,
)


class TestBackup(unittest.TestCase):
    """Test cases for the backup utility."""

    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for the test
        self.temp_dir = tempfile.mkdtemp()
        self.backup_path = Path(self.temp_dir) / "backups"
        self.backup_path.mkdir(exist_ok=True)
        self.timestamp = create_timestamp()
        
        # Create a mock conversations directory
        self.mock_conversations_dir = Path(self.temp_dir) / "src" / "data" / "conversations"
        self.mock_conversations_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a mock conversation file
        mock_conversation = self.mock_conversations_dir / "test_conversation.json"
        with open(mock_conversation, "w") as f:
            f.write('{"test": "data"}')
        
        # Create a mock chroma directory
        self.mock_chroma_dir = Path(self.temp_dir) / "src" / "data" / "chroma"
        self.mock_chroma_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a mock chroma database file
        mock_chroma_db = self.mock_chroma_dir / "chroma.sqlite3"
        with open(mock_chroma_db, "w") as f:
            f.write("mock sqlite data")
        
        # Save the current working directory
        self.original_cwd = os.getcwd()
        
        # Change to the temporary directory
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up the test environment."""
        # Change back to the original working directory
        os.chdir(self.original_cwd)
        
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_create_backup_directory(self):
        """Test creating a backup directory."""
        backup_dir = "test_backups"
        backup_path = create_backup_directory(backup_dir)
        self.assertTrue(backup_path.exists())
        self.assertEqual(backup_path.name, backup_dir)
    
    def test_backup_source_code(self):
        """Test backing up source code."""
        # Create some mock source files
        Path("main.py").write_text("print('Hello, world!')")
        Path("requirements.txt").write_text("langchain==0.1.0")
        
        # Create a mock directory to exclude
        exclude_dir = Path("venv")
        exclude_dir.mkdir(exist_ok=True)
        (exclude_dir / "bin").mkdir(exist_ok=True)
        
        # Backup the source code
        source_backup = backup_source_code(
            self.backup_path, 
            self.timestamp, 
            ["venv"]
        )
        
        # Check that the backup file exists
        self.assertTrue(source_backup.exists())
        self.assertTrue(source_backup.name.startswith("ava6_source_"))
        self.assertTrue(source_backup.name.endswith(".zip"))
    
    def test_backup_conversations(self):
        """Test backing up conversations."""
        # Backup the conversations
        conversations_backup = backup_conversations(
            self.backup_path, 
            self.timestamp
        )
        
        # Check that the backup file exists
        self.assertTrue(conversations_backup.exists())
        self.assertTrue(conversations_backup.name.startswith("ava6_conversations_"))
        self.assertTrue(conversations_backup.name.endswith(".zip"))
    
    def test_backup_vector_database(self):
        """Test backing up the vector database."""
        # Backup the vector database
        chroma_backup = backup_vector_database(
            self.backup_path, 
            self.timestamp
        )
        
        # Check that the backup file exists
        self.assertTrue(chroma_backup.exists())
        self.assertTrue(chroma_backup.name.startswith("ava6_chroma_"))
        self.assertTrue(chroma_backup.name.endswith(".zip"))
    
    def test_create_full_backup(self):
        """Test creating a full backup."""
        # Create individual backups
        source_backup = backup_source_code(
            self.backup_path, 
            self.timestamp, 
            ["venv"]
        )
        
        conversations_backup = backup_conversations(
            self.backup_path, 
            self.timestamp
        )
        
        chroma_backup = backup_vector_database(
            self.backup_path, 
            self.timestamp
        )
        
        # Create a full backup
        full_backup = create_full_backup(
            self.backup_path,
            self.timestamp,
            source_backup,
            conversations_backup,
            chroma_backup
        )
        
        # Check that the full backup file exists
        self.assertTrue(full_backup.exists())
        self.assertTrue(full_backup.name.startswith("ava6_full_backup_"))
        self.assertTrue(full_backup.name.endswith(".zip"))
    
    def test_cleanup_old_backups(self):
        """Test cleaning up old backups."""
        # Create some mock backup files with old timestamps
        old_backup = self.backup_path / "ava6_full_backup_20200101_000000.zip"
        old_backup.write_text("mock backup data")
        
        # Set the modification time to a long time ago
        os.utime(old_backup, (0, 0))
        
        # Create a recent backup file
        recent_backup = self.backup_path / f"ava6_full_backup_{self.timestamp}.zip"
        recent_backup.write_text("mock backup data")
        
        # Clean up old backups
        cleanup_old_backups(self.backup_path, 7)
        
        # Check that the old backup was removed and the recent one remains
        self.assertFalse(old_backup.exists())
        self.assertTrue(recent_backup.exists())


if __name__ == "__main__":
    unittest.main() 