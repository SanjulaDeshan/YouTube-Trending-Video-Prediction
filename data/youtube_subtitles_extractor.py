import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
import time
import logging
import signal
import sys
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for graceful shutdown
interrupted = False
current_df = None
current_output_file = None

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global interrupted, current_df, current_output_file
    
    print("\n" + "="*60)
    print("🛑 INTERRUPTED BY USER (Ctrl+C)")
    print("="*60)
    
    interrupted = True
    
    if current_df is not None and current_output_file is not None:
        try:
            # Save progress so far
            logger.info("💾 Saving progress before exit...")
            current_df.to_csv(current_output_file, index=False)
            logger.info(f"✅ Progress saved to: {current_output_file}")
            
            # Count how many were processed
            processed_count = current_df['video_subtitles'].notna().sum()
            total_count = len(current_df)
            
            print(f"📊 PARTIAL RESULTS SAVED:")
            print(f"   • Videos processed: {processed_count}/{total_count}")
            print(f"   • Success rate: {(processed_count/total_count)*100:.1f}%")
            print(f"   • File saved: {current_output_file}")
            print(f"💡 You can resume processing later by modifying the script")
            
        except Exception as e:
            logger.error(f"❌ Error saving progress: {e}")
    
    print("👋 Exiting gracefully...")
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

def get_video_subtitles(video_id: str, languages=['en', 'en-US', 'en-GB', 'auto']) -> Optional[str]:
    """
    Extract subtitles for a given YouTube video ID with enhanced detection.
    
    Args:
        video_id (str): YouTube video ID
        languages (list): List of language codes to try, in order of preference
    
    Returns:
        str: Concatenated subtitle text or None if not available
    """
    try:
        # First, try to list all available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Get all available transcript languages
        available_transcripts = {}
        for transcript in transcript_list:
            available_transcripts[transcript.language_code] = transcript
            
        logger.debug(f"Available transcripts for {video_id}: {list(available_transcripts.keys())}")
        
        # Try manual transcripts first (higher quality)
        for lang in languages:
            if lang == 'auto':
                continue  # Skip auto for now, try manual first
                
            if lang in available_transcripts:
                transcript = available_transcripts[lang]
                if not transcript.is_generated:  # Manual transcript
                    try:
                        transcript_data = transcript.fetch()
                        subtitle_text = ' '.join([entry['text'] for entry in transcript_data])
                        logger.debug(f"Found manual transcript in {lang} for {video_id}")
                        return subtitle_text.strip()
                    except Exception as e:
                        logger.debug(f"Failed to fetch manual transcript in {lang}: {e}")
                        continue
        
        # Try auto-generated transcripts
        try:
            # Try to find any auto-generated transcript
            for transcript in transcript_list:
                if transcript.is_generated:
                    transcript_data = transcript.fetch()
                    subtitle_text = ' '.join([entry['text'] for entry in transcript_data])
                    logger.debug(f"Found auto-generated transcript in {transcript.language_code} for {video_id}")
                    return subtitle_text.strip()
        except Exception as e:
            logger.debug(f"Failed to get auto-generated transcript: {e}")
        
        # If no transcripts found
        logger.debug(f"No accessible transcripts found for {video_id}")
        return None
        
    except Exception as e:
        # More specific error handling
        error_msg = str(e).lower()
        if "transcript" in error_msg and "disabled" in error_msg:
            logger.debug(f"Transcripts disabled for video {video_id}")
        elif "video" in error_msg and ("unavailable" in error_msg or "private" in error_msg):
            logger.debug(f"Video {video_id} unavailable or private")
        elif "no transcript" in error_msg:
            logger.debug(f"No transcripts available for video {video_id}")
        else:
            logger.debug(f"Unknown error for video {video_id}: {e}")
        return None

def process_youtube_dataset(input_file: str, output_file: str, delay: float = 1.0):
    """
    Process the YouTube dataset and add subtitles column.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        delay (float): Delay between API calls in seconds
    """
    global current_df, current_output_file, interrupted
    
    try:
        # Load the dataset
        logger.info(f"Loading dataset from {input_file}")
        df = pd.read_csv(input_file)
        
        # Set global variables for signal handler
        current_df = df
        current_output_file = output_file
        
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info("💡 Press Ctrl+C to stop and save progress at any time")
        
        # Initialize subtitles column
        df['video_subtitles'] = None
        
        # Process each video
        total_videos = len(df)
        successful_extractions = 0
        
        for index, row in df.iterrows():
            # Check if interrupted
            if interrupted:
                logger.info("🛑 Processing interrupted by user")
                break
                
            video_id = row['ytvideoid']
            video_title = row.get('video_title', 'Unknown')
            
            # Show progress every 10 videos to reduce spam
            if index % 10 == 0 or index < 10:
                logger.info(f"Processing video {index + 1}/{total_videos}: {video_id} - {video_title[:50]}...")
                logger.info(f"📊 Progress: {successful_extractions}/{index + 1} successful ({(successful_extractions/(index + 1))*100:.1f}%)")
            
            # Get subtitles
            subtitles = get_video_subtitles(video_id)
            
            if subtitles:
                df.at[index, 'video_subtitles'] = subtitles
                successful_extractions += 1
                if index % 10 == 0 or successful_extractions < 20:  # Show first 20 successes
                    logger.info(f"✅ SUCCESS: {video_id} - {len(subtitles)} characters")
            else:
                df.at[index, 'video_subtitles'] = "No subtitles available"
                if index < 10:  # Only show detailed warnings for first few
                    logger.warning(f"✗ No subtitles: {video_id}")
            
            # Save progress periodically (every 100 videos)
            if (index + 1) % 100 == 0:
                logger.info(f"💾 Saving progress... ({index + 1}/{total_videos} processed)")
                df.to_csv(output_file, index=False)
            
            # Update global reference
            current_df = df
            
            # Add delay to avoid rate limiting (check for interrupt during delay)
            if delay > 0:
                for i in range(int(delay * 10)):  # Check every 0.1 seconds
                    if interrupted:
                        break
                    time.sleep(0.1)
        
        # Save the updated dataset (if not interrupted)
        if not interrupted:
            logger.info(f"Saving updated dataset to {output_file}")
            df.to_csv(output_file, index=False)
            
            # Print summary
            logger.info(f"\n{'='*50}")
            logger.info(f"PROCESSING COMPLETE")
            logger.info(f"{'='*50}")
            logger.info(f"Total videos processed: {total_videos}")
            logger.info(f"Successful subtitle extractions: {successful_extractions}")
            logger.info(f"Success rate: {(successful_extractions/total_videos)*100:.1f}%")
            logger.info(f"Output saved to: {output_file}")
        
    except FileNotFoundError:
        logger.error(f"Input file '{input_file}' not found.")
    except KeyboardInterrupt:
        # This shouldn't happen now due to signal handler, but just in case
        logger.info("🛑 Process interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        # Try to save progress even on error
        if current_df is not None and current_output_file is not None:
            try:
                current_df.to_csv(current_output_file, index=False)
                logger.info(f"💾 Progress saved to: {current_output_file}")
            except:
                pass

def main():
    # Configuration
    INPUT_FILE = "entertainment_videos.csv"
    OUTPUT_FILE = "entertainment_videos_with_subtitles.csv"
    DELAY_BETWEEN_REQUESTS = 1.0  # seconds
    
    logger.info("Starting YouTube Subtitles Extractor")
    logger.info(f"Input file: {INPUT_FILE}")
    logger.info(f"Output file: {OUTPUT_FILE}")
    logger.info(f"Delay between requests: {DELAY_BETWEEN_REQUESTS} seconds")
    
    # Process the dataset
    process_youtube_dataset(INPUT_FILE, OUTPUT_FILE, DELAY_BETWEEN_REQUESTS)

if __name__ == "__main__":
    # First, install required packages
    print("Make sure you have installed the required packages:")
    print("pip install pandas youtube-transcript-api")
    print("\nStarting processing...")
    
    main()