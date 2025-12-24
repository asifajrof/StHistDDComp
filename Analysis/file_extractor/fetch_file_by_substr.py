import glob
import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class SearchConfig(BaseModel):
    """Simple configuration for file search."""
    root_dir: str = Field(..., description="Root directory to search in")
    substrings: List[str] = Field(..., description="Substrings to search for in file/directory names")
    file_extension: Optional[str] = Field(default=None, description="File extension filter (e.g., '.csv')")
    case_sensitive: bool = Field(default=True, description="Whether substring matching should be case sensitive")


class FileSearcher:
    """Simple file searcher using glob patterns."""
    
    def __init__(self, config: SearchConfig):
        """Initialize with search configuration."""
        self.config = config
    
    def search(self) -> List[str]:
        """
        Search for files matching the configuration.
        
        Returns:
            List of file paths that contain any of the substrings
        """
        root_path = Path(self.config.root_dir)
        
        if not root_path.exists():
            raise ValueError(f"Directory {root_path} does not exist")
        
        # Build glob pattern
        if self.config.file_extension:
            pattern = f"**/*{self.config.file_extension}"
        else:
            pattern = "**/*"
        
        search_path = root_path / pattern
        all_files = glob.glob(str(search_path), recursive=True)
        
        # Filter files that contain any substring
        matching_files = []
        for file_path in all_files:
            if os.path.isfile(file_path):  
                file_path_cased = file_path              
                # Handle case sensitivity
                if not self.config.case_sensitive:
                    file_path_cased = file_path.lower()
                
                # Check if any substring is in the filename
                for substring in self.config.substrings:
                    search_substring = substring.lower() if not self.config.case_sensitive else substring
                    if search_substring in file_path_cased:
                        matching_files.append(file_path)
                        break  # Found one substring, move to next file
        
        return sorted(matching_files)
    
    def search_common(self) -> List[str]:
        """
        Search for files that contain ALL substrings.
        
        Returns:
            List of file paths that contain all substrings
        """
        root_path = Path(self.config.root_dir)
        
        if not root_path.exists():
            raise ValueError(f"Directory {root_path} does not exist")
        
        # Build glob pattern
        if self.config.file_extension:
            pattern = f"**/*{self.config.file_extension}"
        else:
            pattern = "**/*"
        
        search_path = root_path / pattern
        all_files = glob.glob(str(search_path), recursive=True)
        
        # Filter files that contain ALL substrings
        matching_files = []
        for file_path in all_files:
            if os.path.isfile(file_path):
                file_path_cased = file_path
                # Handle case sensitivity
                if not self.config.case_sensitive:
                    file_path_cased = file_path.lower()
                    search_substrings = [s.lower() for s in self.config.substrings]
                else:
                    search_substrings = self.config.substrings
                
                # Check if ALL substrings are in the file path
                if all(substring in file_path_cased for substring in search_substrings):
                    matching_files.append(file_path)
        
        return sorted(matching_files)

if __name__ == "__main__":
    # Example usage
    config = SearchConfig(
        root_dir="data/manual_annotations/",
        substrings=["hbc"],
        file_extension=".csv",
        case_sensitive=False
    )
    searcher = FileSearcher(config)

    files_with_all = searcher.search_common()
    print("\nFiles containing all of the substrings:")
    for f in files_with_all:
        print(f)

