import os
from decouple import config
from openai import OpenAI
from sqlalchemy import (
    create_engine,
    BigInteger, Column, Integer, String, DateTime, ForeignKey, UniqueConstraint, Text, text, func, Index
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector

# --- 1. Database Setup and Models ---
# Load environment variables from .env file
DATABASE_URL = config("DATABASE_URL", default="postgresql://admin_user:asdf@localhost:5432/test_vector_db")
OPENAI_API_KEY = config("OPENAI_API_KEY")

# OpenAI client setup
client = OpenAI(api_key=OPENAI_API_KEY)
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # Dimension for text-embedding-3-small

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# The models from your prompt
class Channel(Base):
    __tablename__ = "channel"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    youtube_channel_id = Column(String(255), unique=True, nullable=False, index=True)
    title = Column(String(512), nullable=True)
    description = Column(String(5000), nullable=True)
    published_at = Column(DateTime, nullable=True)
    view_count = Column(BigInteger, nullable=True)
    subscriber_count = Column(BigInteger, nullable=True)
    video_count = Column(BigInteger, nullable=True)
    country = Column(String(2), nullable=True)
    handle = Column(String(255), nullable=True)
    url = Column(String(255), nullable=True)
    created_at = Column(DateTime, server_default=text("now()"))
    videos = relationship("Video", back_populates="channel")


class Video(Base):
    __tablename__ = "video"
    id = Column(Integer, primary_key=True, autoincrement=True)
    youtube_video_id = Column(String(255), nullable=False)
    channel_id_fk = Column(Integer, ForeignKey("channel.id", ondelete="CASCADE"), nullable=False)
    inserted_at = Column(DateTime, server_default=text("now()"))
    transcription = Column(String(1000), nullable=True)
    channel = relationship("Channel", back_populates="videos")
    transcripts = relationship("VideoTranscript", back_populates="video", lazy=True)
    transcription_done_at = Column(DateTime, nullable=True)
    __table_args__ = (UniqueConstraint("youtube_video_id", "channel_id_fk", name="uq_video_channel"),)


class VideoTranscript(Base):
    __tablename__ = "video_transcript"
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id_fk = Column(Integer, ForeignKey("video.id", ondelete="CASCADE"), nullable=False)
    transcription = Column(Text, nullable=True)
    transcription_tsv = Column(TSVECTOR, nullable=True)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=True)
    created_at = Column(DateTime, server_default=text("now()"))
    video = relationship("Video", back_populates="transcripts")

    __table_args__ = (
        # Index 1: GIN index for fast full-text search performance.
        Index('idx_video_transcript_tsv', 'transcription_tsv', postgresql_using='gin'),

        # Index 2: HNSW index for fast vector similarity search (ANN).
        # This is crucial for making vector searches performant on large datasets.
        # 'vector_cosine_ops' is specified because we are using the cosine
        # distance operator (<=>) in our semantic search query.
        Index(
            'idx_hnsw_embedding',
            'embedding',
            postgresql_using='hnsw',
            postgresql_ops={'embedding': 'vector_cosine_ops'}
        ),
    )


# --- 2. Main Proof-of-Concept Script ---
def get_embedding(text, model=EMBEDDING_MODEL):
    """Generates an embedding for a given text using OpenAI API."""
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def run_poc():
    """
    A self-contained script to demonstrate tsvector and pgvector functionality.
    """
    # Step 1: Enable vector extension and run initial migrations.
    print("--- Enabling vector extension and running migrations (if necessary) ---")
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)
    print("Migrations complete.")
    print("Ensured GIN index on 'transcription_tsv' and HNSW index on 'embedding' exist.\n")


    db = SessionLocal()

    try:
        # Step 2: Populate with fake data if the database is empty.
        if db.query(VideoTranscript).first() is None:
            print("--- Database is empty. Populating with sample data ---")

            # Create parent records to satisfy foreign keys
            sample_channel = Channel(youtube_channel_id="UC_poc_channel", title="Test Channel")
            db.add(sample_channel)
            db.flush()  # Flush to get the channel ID

            sample_video = Video(youtube_video_id="dQw4w9WgXcQ", channel_id_fk=sample_channel.id)
            db.add(sample_video)
            db.flush()  # Flush to get the video ID

            # Sample Portuguese sentences
            transcripts_data = [
                "O rápido cão marrom salta sobre o cão preguiçoso.",
                "A busca por conhecimento é uma jornada sem fim.",
                "A tecnologia transforma a sociedade de maneiras profundas.",
                "O cão é o melhor amigo do homem, e os cães são leais.",
                "A busca por vida em outros planetas continua.",
            ]

            print("Generating embeddings for sample data...")
            for text_content in transcripts_data:
                # Generate embedding for the transcription
                embedding_vector = get_embedding(text_content)

                transcript = VideoTranscript(
                    video_id_fk=sample_video.id,
                    transcription=text_content,
                    # The tsvector is generated here using the 'portuguese' dictionary
                    transcription_tsv=func.to_tsvector('portuguese', text_content),
                    embedding=embedding_vector
                )
                db.add(transcript)

            db.commit()
            print("Sample data inserted, tsvector and embedding columns populated.\n")
        else:
            print("--- Database already contains data. Skipping population. ---\n")


        # --- 3. Performing Full-Text Searches (TSVector) ---
        print("--- Performing Full-Text Searches (TSVector) ---")

        # Example 1: Simple word search
        # Searches for the root form of 'cão' (dog).
        print("--- 1. Simple search for 'cão' ---")
        query = func.to_tsquery('portuguese', 'cão')
        results = db.query(VideoTranscript).filter(VideoTranscript.transcription_tsv.op('@@')(query)).all()
        for r in results:
            print(f"  - Found in: '{r.transcription}'")

        # Example 2: Stemming demonstration
        # The text contains 'salta' (jumps), but we search for 'saltar' (to jump).
        # PostgreSQL's stemming finds the match.
        print("\n--- 2. Stemming: Searching for 'saltar' (finds 'salta') ---")
        query = func.to_tsquery('portuguese', 'saltar')
        results = db.query(VideoTranscript).filter(VideoTranscript.transcription_tsv.op('@@')(query)).all()
        for r in results:
            print(f"  - Found in: '{r.transcription}'")

        # Example 3: AND operator
        # Finds documents containing both 'cão' AND 'marrom'.
        print("\n--- 3. AND search: 'cão' AND 'marrom' ---")
        query = func.to_tsquery('portuguese', 'cão & marrom')
        results = db.query(VideoTranscript).filter(VideoTranscript.transcription_tsv.op('@@')(query)).all()
        for r in results:
            print(f"  - Found in: '{r.transcription}'")

        # Example 4: OR operator
        # Finds documents containing either 'conhecimento' OR 'tecnologia'.
        print("\n--- 4. OR search: 'conhecimento' OR 'tecnologia' ---")
        query = func.to_tsquery('portuguese', 'conhecimento | tecnologia')
        results = db.query(VideoTranscript).filter(VideoTranscript.transcription_tsv.op('@@')(query)).all()
        for r in results:
            print(f"  - Found in: '{r.transcription}'")

        # Example 5: Prefix search
        # Finds words that start with 'tecno'.
        print("\n--- 5. Prefix search: 'tecno:*' ---")
        query = func.to_tsquery('portuguese', 'tecno:*')
        results = db.query(VideoTranscript).filter(VideoTranscript.transcription_tsv.op('@@')(query)).all()
        for r in results:
            print(f"  - Found in: '{r.transcription}'")

        # Example 6: Phrase search
        # Finds 'cão' immediately followed by 'preguiçoso'.
        print("\n--- 6. Phrase search: 'cão preguiçoso' ---")
        query = func.to_tsquery('portuguese', 'cão <-> preguiçoso')
        results = db.query(VideoTranscript).filter(VideoTranscript.transcription_tsv.op('@@')(query)).all()
        for r in results:
            print(f"  - Found in: '{r.transcription}'")

        # Example 7: Search with Ranking
        # Ranks the results based on how relevant they are to the query "busca".
        # `ts_rank` considers word frequency and proximity.
        print("\n--- 7. Search with Ranking for 'busca' ---")
        search_term = 'busca'
        query = func.to_tsquery('portuguese', search_term)
        rank = func.ts_rank(VideoTranscript.transcription_tsv, query).label('rank')
        results = db.query(VideoTranscript, rank)\
            .filter(VideoTranscript.transcription_tsv.op('@@')(query))\
            .order_by(rank.desc())\
            .all()
        for r, rank_score in results:
            print(f"  - Rank: {rank_score:.4f}, Text: '{r.transcription}'")

        # --- 4. Performing Semantic Similarity Search (PGVector) ---
        print("\n--- Performing Semantic Similarity Search (PGVector) ---")

        # Example 8: Semantic search
        # We'll search for a concept semantically similar to one of our sentences.
        # "vida em outros planetas" is semantically close to "vida em outros mundos".
        search_query = "vida em outros mundos"
        print(f"\n--- 8. Semantic search for: '{search_query}' ---")

        # Generate embedding for the search query
        search_embedding = get_embedding(search_query)

        # Perform cosine similarity search
        # The `<=>` operator calculates the cosine distance (0=exact match, 1=opposite, 2=orthogonal)
        # We order by this distance to get the closest matches first.
        # This query will be fast now because of the HNSW index.
        results = db.query(VideoTranscript).order_by(
            VideoTranscript.embedding.cosine_distance(search_embedding)
        ).limit(3).all()

        print("Top 3 most semantically similar results:")
        for r in results:
            # You can also calculate the similarity score (1 - distance) if you prefer
            # For this example, we'll just show the text.
            print(f"  - Found: '{r.transcription}'")

    finally:
        # --- 5. Cleanup ---
        # To persist data across runs, we'll comment out the drop_all call.
        # print("\n--- Cleaning up (dropping tables) ---")
        # Base.metadata.drop_all(bind=engine)
        db.close()
        print("\nDone.")


if __name__ == "__main__":
    run_poc()
