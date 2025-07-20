import pandas as pd
import requests
import time
import json
import os
from typing import List, Dict, Optional
import logging
import pickle
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YouTubeDataEnricher:
    def __init__(self, api_key: str, checkpoint_dir: str = "checkpoints"):
        """
        Initialize the YouTube Data Enricher with checkpoint support
        
        Args:
            api_key (str): YouTube Data API v3 key
            checkpoint_dir (str): Directory to store checkpoint files
        """
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3/videos"
        self.captions_url = "https://www.googleapis.com/youtube/v3/captions"
        self.batch_size = 50
        self.rate_limit_delay = 0.1
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, data: Dict, checkpoint_file: str):
        """Save checkpoint data to file"""
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_file: str) -> Optional[Dict]:
        """Load checkpoint data from file"""
        try:
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Checkpoint loaded: {checkpoint_file}")
                return data
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
        return None
    
    def get_category_name(self, category_id: str) -> str:
        """
        Convert YouTube category ID to category name
        
        Args:
            category_id (str): YouTube category ID
            
        Returns:
            str: Category name
        """
        # YouTube category mapping (most common categories)
        category_map = {
            '1': 'Film & Animation',
            '2': 'Autos & Vehicles', 
            '10': 'Music',
            '15': 'Pets & Animals',
            '17': 'Sports',
            '18': 'Short Movies',
            '19': 'Travel & Events',
            '20': 'Gaming',
            '21': 'Videoblogging',
            '22': 'People & Blogs',
            '23': 'Comedy',
            '24': 'Entertainment',
            '25': 'News & Politics',
            '26': 'Howto & Style',
            '27': 'Education',
            '28': 'Science & Technology',
            '29': 'Nonprofits & Activism',
            '30': 'Movies',
            '31': 'Anime/Animation',
            '32': 'Action/Adventure',
            '33': 'Classics',
            '34': 'Comedy',
            '35': 'Documentary',
            '36': 'Drama',
            '37': 'Family',
            '38': 'Foreign',
            '39': 'Horror',
            '40': 'Sci-Fi/Fantasy',
            '41': 'Thriller',
            '42': 'Shorts',
            '43': 'Shows',
            '44': 'Trailers'
        }
        
        return category_map.get(str(category_id), f'Unknown Category ({category_id})')

    def save_progress_csv(self, df: pd.DataFrame, output_file: str):
        """Save current progress to CSV file"""
        try:
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = output_file.replace('.csv', f'_backup_{timestamp}.csv')
            
            df.to_csv(backup_file, index=False)
            df.to_csv(output_file, index=False)
            logger.info(f"Progress saved to: {output_file} (backup: {backup_file})")
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def get_video_details(self, video_ids: List[str]) -> Dict[str, Dict]:
        """
        Fetch video details from YouTube API for a batch of video IDs
        """
        if not video_ids:
            return {}
            
        ids_str = ','.join(video_ids)
        
        params = {
            'part': 'snippet,contentDetails',
            'id': ids_str,
            'key': self.api_key
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                # Handle API quota exceeded
                if 'error' in data:
                    if data['error'].get('code') == 403:
                        logger.error("API quota exceeded. Please wait or use a different API key.")
                        raise Exception("API quota exceeded")
                
                video_details = {}
                for item in data.get('items', []):
                    video_id = item['id']
                    snippet = item.get('snippet', {})
                    
                    # Get category name from category ID
                    category_id = snippet.get('categoryId', '')
                    category_name = self.get_category_name(category_id)
                    
                    video_details[video_id] = {
                        'title': snippet.get('title', ''),
                        'description': snippet.get('description', ''),
                        'category': category_name,
                    }
                    
                return video_details
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error("Max retries reached for video details")
                    return {}
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return {}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse API response: {e}")
                return {}
        
        return {}
    
    def get_caption_info(self, video_id: str) -> str:
        """
        Get caption availability for a single video
        """
        params = {
            'part': 'snippet',
            'videoId': video_id,
            'key': self.api_key
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(self.captions_url, params=params, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('items'):
                        return 'Available'
                    else:
                        return 'Not Available'
                elif response.status_code == 403:
                    # API quota or permissions issue
                    return 'Unknown (API limit)'
                else:
                    return 'Unknown'
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Caption request timeout for {video_id} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    return 'Unknown (timeout)'
            except Exception as e:
                logger.warning(f"Failed to get captions for {video_id}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    return 'Unknown (error)'
        
        return 'Unknown'
    
    def process_csv(self, input_file: str, output_file: str, video_id_column: str = 'ytvideoid'):
        """
        Process the CSV file with checkpoint support
        """
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{os.path.basename(input_file)}_checkpoint.pkl")
        
        logger.info(f"Loading CSV file: {input_file}")
        
        # Read the CSV file
        try:
            df = pd.read_csv(input_file)
        except Exception as e:
            logger.error(f"Failed to read CSV file: {e}")
            return
        
        logger.info(f"Loaded {len(df)} records")
        logger.info(f"Existing columns: {list(df.columns)}")
        
        # Check if video ID column exists
        if video_id_column not in df.columns:
            logger.error(f"Column '{video_id_column}' not found. Available: {list(df.columns)}")
            return
        
        # Initialize new columns if they don't exist
        if 'video_title' not in df.columns:
            df['video_title'] = ''
        if 'video_description' not in df.columns:
            df['video_description'] = ''
        if 'video_caption' not in df.columns:
            df['video_caption'] = ''
        if 'video_category' not in df.columns:
            df['video_category'] = ''
        
        # Load checkpoint if exists
        checkpoint_data = self.load_checkpoint(checkpoint_file)
        processed_videos = set()
        processed_captions = set()
        
        if checkpoint_data:
            processed_videos = set(checkpoint_data.get('processed_videos', []))
            processed_captions = set(checkpoint_data.get('processed_captions', []))
            logger.info(f"Resuming: {len(processed_videos)} videos processed, {len(processed_captions)} captions processed")
        
        # Get unique video IDs
        video_ids = df[video_id_column].dropna().astype(str).unique().tolist()
        remaining_videos = [vid for vid in video_ids if vid not in processed_videos]
        
        logger.info(f"Total unique videos: {len(video_ids)}")
        logger.info(f"Remaining to process: {len(remaining_videos)}")
        
        # Phase 1: Process video details in batches
        if remaining_videos:
            logger.info("=== PHASE 1: Fetching video titles and descriptions ===")
            
            total_batches = (len(remaining_videos) + self.batch_size - 1) // self.batch_size
            
            for i in range(0, len(remaining_videos), self.batch_size):
                batch_num = (i // self.batch_size) + 1
                batch_ids = remaining_videos[i:i + self.batch_size]
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_ids)} videos)")
                
                try:
                    # Get video details for this batch
                    batch_details = self.get_video_details(batch_ids)
                    
                    # Update dataframe
                    for video_id, details in batch_details.items():
                        mask = df[video_id_column].astype(str) == video_id
                        df.loc[mask, 'video_title'] = details['title']
                        df.loc[mask, 'video_description'] = details['description']
                        df.loc[mask, 'video_category'] = details['category']
                        processed_videos.add(video_id)
                    
                    # Save progress every 5 batches
                    if batch_num % 5 == 0:
                        self.save_progress_csv(df, output_file)
                        
                        # Save checkpoint
                        checkpoint_data = {
                            'processed_videos': list(processed_videos),
                            'processed_captions': list(processed_captions),
                            'last_batch': batch_num,
                            'timestamp': datetime.now().isoformat()
                        }
                        self.save_checkpoint(checkpoint_data, checkpoint_file)
                    
                    # Rate limiting
                    time.sleep(self.rate_limit_delay)
                    
                except KeyboardInterrupt:
                    logger.info("Process interrupted by user. Saving progress...")
                    self.save_progress_csv(df, output_file)
                    
                    checkpoint_data = {
                        'processed_videos': list(processed_videos),
                        'processed_captions': list(processed_captions),
                        'last_batch': batch_num,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.save_checkpoint(checkpoint_data, checkpoint_file)
                    logger.info("Progress saved. You can resume by running the script again.")
                    return
                except Exception as e:
                    logger.error(f"Error in batch {batch_num}: {e}")
                    # Continue with next batch
                    continue
        
        # Phase 2: Process captions
        remaining_for_captions = [vid for vid in video_ids if vid not in processed_captions]
        
        if remaining_for_captions:
            logger.info("=== PHASE 2: Fetching caption information ===")
            
            total_captions = len(remaining_for_captions)
            
            for idx, video_id in enumerate(remaining_for_captions, 1):
                try:
                    logger.info(f"Processing captions {idx}/{total_captions}: {video_id}")
                    
                    caption_info = self.get_caption_info(video_id)
                    
                    # Update dataframe
                    mask = df[video_id_column].astype(str) == video_id
                    df.loc[mask, 'video_caption'] = caption_info
                    processed_captions.add(video_id)
                    
                    # Save progress every 100 videos
                    if idx % 100 == 0:
                        self.save_progress_csv(df, output_file)
                        
                        checkpoint_data = {
                            'processed_videos': list(processed_videos),
                            'processed_captions': list(processed_captions),
                            'caption_progress': idx,
                            'timestamp': datetime.now().isoformat()
                        }
                        self.save_checkpoint(checkpoint_data, checkpoint_file)
                        logger.info(f"Progress checkpoint saved ({idx}/{total_captions})")
                    
                    # Rate limiting for captions
                    time.sleep(0.05)
                    
                except KeyboardInterrupt:
                    logger.info("Process interrupted by user. Saving progress...")
                    self.save_progress_csv(df, output_file)
                    
                    checkpoint_data = {
                        'processed_videos': list(processed_videos),
                        'processed_captions': list(processed_captions),
                        'caption_progress': idx,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.save_checkpoint(checkpoint_data, checkpoint_file)
                    logger.info("Progress saved. You can resume by running the script again.")
                    return
                except Exception as e:
                    logger.error(f"Error processing captions for {video_id}: {e}")
                    # Mark as processed even if failed to avoid infinite retries
                    processed_captions.add(video_id)
                    df.loc[df[video_id_column].astype(str) == video_id, 'video_caption'] = 'Error'
        
        # Final save
        self.save_progress_csv(df, output_file)
        
        # Clean up checkpoint file on successful completion
        try:
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                logger.info("Checkpoint file removed (process completed successfully)")
        except Exception as e:
            logger.warning(f"Could not remove checkpoint file: {e}")
        
        # Print summary
        filled_titles = (df['video_title'] != '').sum()
        filled_descriptions = (df['video_description'] != '').sum()
        filled_captions = (df['video_caption'] != '').sum()
        filled_categories = (df['video_category'] != '').sum()
        
        logger.info("=== FINAL SUMMARY ===")
        logger.info(f"Total records: {len(df)}")
        logger.info(f"Videos with titles: {filled_titles}")
        logger.info(f"Videos with descriptions: {filled_descriptions}")
        logger.info(f"Videos with caption info: {filled_captions}")
        logger.info(f"Videos with categories: {filled_categories}")
        logger.info(f"Output saved to: {output_file}")

def main():
    """
    Main function to run the YouTube data enricher
    """
    # Get API key from environment variable
    API_KEY = os.getenv('YOUTUBE_API_KEY')
    
    if not API_KEY:
        print("ERROR: YOUTUBE_API_KEY not found in environment variables.")
        print("Please create a .env file in your project root with:")
        print("YOUTUBE_API_KEY=your_actual_api_key_here")
        return
    
    # Configuration - Update these paths for your project
    INPUT_CSV = "data/youtube_data.csv"      # Update with your CSV file name
    OUTPUT_CSV = "data/enriched_youtube_data.csv" # Output file name
    VIDEO_ID_COLUMN = "ytvideoid"                 # Column name containing video IDs
    
    # Check if input file exists
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: Input file '{INPUT_CSV}' not found.")
        print("Please make sure your CSV file is in the correct location.")
        return
    
    # Create enricher instance
    enricher = YouTubeDataEnricher(API_KEY)
    
    # Process the CSV file
    enricher.process_csv(INPUT_CSV, OUTPUT_CSV, VIDEO_ID_COLUMN)

if __name__ == "__main__":
    main()

# Alternative quick setup function with resume capability
def quick_enrich_with_resume(csv_file: str, output_file: str = None):
    """
    Quick function to enrich CSV with video data (supports resume)
    Uses API key from environment variables
    
    Args:
        csv_file (str): Path to CSV file
        output_file (str): Path to output file (optional)
    """
    API_KEY = os.getenv('YOUTUBE_API_KEY')
    
    if not API_KEY:
        print("ERROR: YOUTUBE_API_KEY not found in environment variables.")
        return None
    
    if output_file is None:
        output_file = csv_file.replace('.csv', '_enriched.csv')
    
    enricher = YouTubeDataEnricher(API_KEY)
    enricher.process_csv(csv_file, output_file)
    
    return output_file

# Usage examples:
# python youtube_enricher.py
# OR in Python script:
# enriched_file = quick_enrich_with_resume("data/youtube_data.csv")