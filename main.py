import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Now import your custom modules
from src.ingest import MusicDataIngestor
from src.pipeline import MusicRAGPipeline

def main():
    # Your code stays exactly the same!
    RE_INGEST_DATA = True 
    
    if RE_INGEST_DATA:
        ingestor = MusicDataIngestor()
        ingestor.run_ingestion()
        print("-" * 50)

    pipeline = MusicRAGPipeline()
    print("-" * 50)
    
    query_str = "I want a song that features soulful, jazz-heavy production with a nostalgic late-night vibe. What do you recommend?"
    response = pipeline.query(query_str)
    
    print("\n--- AI Musicologist Recommendation ---")
    print(response)

if __name__ == "__main__":
    main()