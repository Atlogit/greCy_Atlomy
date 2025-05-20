#!/usr/bin/env python3
"""
File I/O utilities for handling various file formats and temporary directories.
"""

import logging
import tempfile
import zipfile
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import spacy
from spacy.tokens import Doc, DocBin

logger = logging.getLogger(__name__)

class FileIO:
    """Handles file input/output operations and temporary directory management."""
    
    @staticmethod
    def read_json(file_path: Path) -> Dict:
        """
        Read and parse a JSON file.
        
        Args:
            file_path (Path): Path to the JSON file
            
        Returns:
            Dict: Parsed JSON content
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            raise

    @staticmethod
    def write_json(data: Dict, file_path: Path, indent: int = 2) -> None:
        """
        Write data to a JSON file.
        
        Args:
            data (Dict): Data to write
            file_path (Path): Path to write the JSON file
            indent (int): Number of spaces for indentation
            
        Raises:
            IOError: If writing fails
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
        except IOError as e:
            logger.error(f"Failed to write to {file_path}: {e}")
            raise

    @staticmethod
    def read_text_file(file_path: Path) -> str:
        """
        Read content from a text file.
        
        Args:
            file_path (Path): Path to the text file
            
        Returns:
            str: File content
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except UnicodeDecodeError as e:
            logger.error(f"Unicode decode error in {file_path}: {e}")
            raise

    @staticmethod
    def write_text_file(content: str, file_path: Path) -> None:
        """
        Write content to a text file.
        
        Args:
            content (str): Content to write
            file_path (Path): Path to write the file
            
        Raises:
            IOError: If writing fails
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except IOError as e:
            logger.error(f"Failed to write to {file_path}: {e}")
            raise

    @staticmethod
    def extract_zip(zip_path: Path, extract_path: Optional[Path] = None) -> Path:
        """
        Extract a zip file to a directory.
        
        Args:
            zip_path (Path): Path to the zip file
            extract_path (Optional[Path]): Path to extract to. If None, creates a temp directory
            
        Returns:
            Path: Path where files were extracted
            
        Raises:
            FileNotFoundError: If zip file doesn't exist
            zipfile.BadZipFile: If file is not a valid zip
        """
        if extract_path is None:
            extract_path = Path(tempfile.mkdtemp())
            
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            return extract_path
        except FileNotFoundError:
            logger.error(f"Zip file not found: {zip_path}")
            raise
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid zip file {zip_path}: {e}")
            raise

    @staticmethod
    def cleanup_directory(directory: Path) -> None:
        """
        Remove a directory and all its contents.
        
        Args:
            directory (Path): Directory to remove
        """
        try:
            for item in directory.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    FileIO.cleanup_directory(item)
            directory.rmdir()
        except Exception as e:
            logger.error(f"Error cleaning up directory {directory}: {e}")
            raise

    @staticmethod
    def save_docs(docs: List[Doc], output_path: Path, store_user_data: bool = True) -> None:
        """
        Save spaCy docs to disk using DocBin.
        
        Args:
            docs (List[Doc]): List of spaCy Doc objects to save
            output_path (Path): Path to save the docs
            store_user_data (bool): Whether to store custom user data
            
        Raises:
            IOError: If saving fails
        """
        try:
            doc_bin = DocBin(docs=docs, store_user_data=store_user_data)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            doc_bin.to_disk(output_path)
            logger.info(f"Saved {len(docs)} documents to {output_path}")
        except Exception as e:
            logger.error(f"Error saving docs to {output_path}: {e}")
            raise

    @staticmethod
    def load_docs(file_path: Path, nlp: Any) -> List[Doc]:
        """
        Load spaCy docs from disk.
        
        Args:
            file_path (Path): Path to the .spacy file
            nlp: Loaded spaCy model/vocab
            
        Returns:
            List[Doc]: List of loaded Doc objects
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        try:
            doc_bin = DocBin().from_disk(file_path)
            return list(doc_bin.get_docs(nlp.vocab))
        except FileNotFoundError:
            logger.error(f"Spacy file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading docs from {file_path}: {e}")
            raise

    @staticmethod
    def ensure_dir(directory: Path) -> None:
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            directory (Path): Directory path to ensure exists
        """
        directory.mkdir(parents=True, exist_ok=True)