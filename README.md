album_review_rag/
│
├── data/                        # Your folder with all the .txt, pdf reviews
│   ├── black_star.txt
│   ├── blonde.txt
│   └── ...
│
├── storage/                     # Where your local FAISS index will be saved
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Stores paths and model names
│   ├── ingest.py                # Class for loading, chunking, & saving vectors
│   └── pipeline.py              # Class for loading the index & generating answers
│
└── main.py                      # The entry point to run queries