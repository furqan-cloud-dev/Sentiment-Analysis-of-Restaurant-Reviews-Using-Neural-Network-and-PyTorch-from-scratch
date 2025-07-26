# requirements.txt
"""
fastapi==0.104.1
uvicorn==0.24.0
sqlalchemy==2.0.23
asyncpg==0.29.0
psycopg2-binary==2.9.9
alembic==1.13.1
pydantic==2.5.0
torch==2.1.0
torchvision==0.16.0
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.24.3
celery==5.3.4
redis==5.0.1
python-multipart==0.0.6
python-dotenv==1.0.0
apscheduler==3.10.4
mlflow==2.8.1
boto3==1.34.0
"""

# .env
"""
DATABASE_URL=postgresql://username:password@localhost/sentiment_db
REDIS_URL=redis://localhost:6379
MLFLOW_TRACKING_URI=http://localhost:5000
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET_NAME=your_model_bucket
"""

# database/models.py
from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    source = Column(String(100), nullable=False)  # e.g., 'google', 'yelp', 'internal'
    rating = Column(Integer, nullable=True)  # 1-5 stars if available
    sentiment_label = Column(Integer, nullable=True)  # 0: negative, 1: neutral, 2: positive, 3: constructive
    confidence_score = Column(Float, nullable=True)
    is_labeled = Column(Boolean, default=False)
    is_training_data = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    processed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index('idx_reviews_created_at', 'created_at'),
        Index('idx_reviews_is_labeled', 'is_labeled'),
        Index('idx_reviews_sentiment_label', 'sentiment_label'),
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String(50), unique=True, nullable=False)
    model_path = Column(String(255), nullable=False)
    accuracy = Column(Float, nullable=True)
    training_samples = Column(Integer, nullable=False)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    training_config = Column(Text, nullable=True)  # JSON string


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(100), unique=True, nullable=False)
    status = Column(String(50), default='pending')  # pending, running, completed, failed
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    training_samples = Column(Integer, nullable=True)
    validation_accuracy = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    model_version = Column(String(50), nullable=True)


# database/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Sync engine for migrations
sync_engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)

# Async engine for FastAPI
async_engine = create_async_engine(ASYNC_DATABASE_URL)
AsyncSessionLocal = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# ml/model.py - Enhanced version of the sentiment model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import pickle
import json
from typing import List, Dict, Tuple
import mlflow
import mlflow.pytorch
from datetime import datetime
import os


class SentimentModel:
    def __init__(self, config: dict = None):
        self.config = config or self.default_config()
        self.model = None
        self.preprocessor = None
        self.label_names = ['Negative', 'Neutral', 'Positive', 'Constructive']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def default_config(self):
        return {
            'embedding_dim': 128,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 15,
            'max_length': 100,
            'min_vocab_freq': 2
        }

    def build_model(self, vocab_size):
        """Build the LSTM model"""

        class SentimentLSTM(nn.Module):
            def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
                super(SentimentLSTM, self).__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                                    batch_first=True, dropout=dropout, bidirectional=True)
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_dim * 2, output_dim)

            def forward(self, x):
                embedded = self.embedding(x)
                lstm_out, (hidden, cell) = self.lstm(embedded)
                hidden = hidden[-2:, :, :]
                hidden = torch.cat([hidden[0], hidden[1]], dim=1)
                output = self.dropout(hidden)
                output = self.fc(output)
                return output

        self.model = SentimentLSTM(
            vocab_size=vocab_size,
            embedding_dim=self.config['embedding_dim'],
            hidden_dim=self.config['hidden_dim'],
            output_dim=4,
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        )
        self.model.to(self.device)
        return self.model

    def train(self, train_data: List[Dict], validation_data: List[Dict] = None):
        """Train the model with tracking"""

        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.config)

            # Prepare data
            texts = [item['text'] for item in train_data]
            labels = [item['label'] for item in train_data]

            # Build preprocessor
            from ml.preprocessing import TextPreprocessor
            self.preprocessor = TextPreprocessor()
            self.preprocessor.build_vocabulary(texts, self.config['min_vocab_freq'])

            # Build model
            self.build_model(self.preprocessor.vocab_size)

            # Create datasets
            train_dataset = SentimentDataset(texts, labels, self.preprocessor, self.config['max_length'])

            if validation_data:
                val_texts = [item['text'] for item in validation_data]
                val_labels = [item['label'] for item in validation_data]
                val_dataset = SentimentDataset(val_texts, val_labels, self.preprocessor, self.config['max_length'])
            else:
                # Split training data
                train_texts, val_texts, train_labels, val_labels = train_test_split(
                    texts, labels, test_size=0.2, random_state=42, stratify=labels
                )
                train_dataset = SentimentDataset(train_texts, train_labels, self.preprocessor,
                                                 self.config['max_length'])
                val_dataset = SentimentDataset(val_texts, val_labels, self.preprocessor, self.config['max_length'])

            # Create dataloaders
            train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'],
                                      shuffle=True, collate_fn=self.collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'],
                                    shuffle=False, collate_fn=self.collate_fn)

            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

            # Training loop
            best_val_acc = 0.0
            for epoch in range(self.config['epochs']):
                train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)
                val_loss, val_acc = self._validate_epoch(val_loader, criterion)

                # Log metrics
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                }, step=epoch)

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_model(f"best_model_epoch_{epoch}.pth")

            # Log final model
            mlflow.pytorch.log_model(self.model, "model")
            mlflow.log_metric("best_val_accuracy", best_val_acc)

            return {
                'best_validation_accuracy': best_val_acc,
                'training_samples': len(train_data),
                'validation_samples': len(validation_data) if validation_data else len(val_texts)
            }

    def _train_epoch(self, train_loader, criterion, optimizer):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            texts = batch['text'].to(self.device)
            labels = batch['label'].to(self.device)

            optimizer.zero_grad()
            outputs = self.model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return total_loss / len(train_loader), 100 * correct / total

    def _validate_epoch(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                texts = batch['text'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(texts)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return total_loss / len(val_loader), 100 * correct / total

    def predict(self, text: str):
        """Predict sentiment for a single text"""
        if not self.model or not self.preprocessor:
            raise ValueError("Model not loaded. Please load a trained model first.")

        self.model.eval()
        sequence = self.preprocessor.text_to_sequence(text, self.config['max_length'])
        sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)

        with torch.no_grad():
            output = self.model(sequence_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return {
            'predicted_sentiment': self.label_names[predicted_class],
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': {self.label_names[i]: prob.item() for i, prob in enumerate(probabilities[0])}
        }

    def save_model(self, filepath: str):
        """Save model and preprocessor"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'preprocessor': self.preprocessor,
            'config': self.config,
            'label_names': self.label_names
        }, filepath)

    def load_model(self, filepath: str):
        """Load model and preprocessor"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.config = checkpoint['config']
        self.preprocessor = checkpoint['preprocessor']
        self.label_names = checkpoint['label_names']

        self.build_model(self.preprocessor.vocab_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def collate_fn(self, batch):
        """Custom collate function for padding sequences"""
        texts = [item['text'] for item in batch]
        labels = [item['label'] for item in batch]

        texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
        labels = torch.stack(labels)

        return {'text': texts_padded, 'label': labels}


# Copy SentimentDataset and TextPreprocessor classes from the original artifact
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, preprocessor, max_length=100):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        sequence = self.preprocessor.text_to_sequence(text, self.max_length)

        return {
            'text': torch.tensor(sequence, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


# ml/preprocessing.py
import re
from collections import Counter
from typing import List


class TextPreprocessor:
    def __init__(self):
        self.vocab = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0

    def clean_text(self, text):
        """Clean and normalize text"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def tokenize(self, text):
        """Simple tokenization"""
        return text.split()

    def build_vocabulary(self, texts, min_freq=2):
        """Build vocabulary from texts"""
        word_counts = Counter()

        for text in texts:
            cleaned = self.clean_text(text)
            tokens = self.tokenize(cleaned)
            word_counts.update(tokens)

        # Filter by minimum frequency
        filtered_words = [word for word, count in word_counts.items() if count >= min_freq]

        # Create vocabulary mappings
        self.word_to_idx = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3
        }

        for word in filtered_words:
            self.word_to_idx[word] = len(self.word_to_idx)

        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)

        return self.word_to_idx

    def text_to_sequence(self, text, max_length=100):
        """Convert text to sequence of indices"""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)

        sequence = [self.word_to_idx.get(token, self.word_to_idx['<UNK>']) for token in tokens]

        # Truncate or pad
        if len(sequence) > max_length:
            sequence = sequence[:max_length]

        return sequence


# ml/training_pipeline.py
import asyncio
from typing import List, Dict
from sqlalchemy import select
from database.database import AsyncSessionLocal
from database.models import Review, ModelVersion, TrainingJob
from ml.model import SentimentModel
import uuid
from datetime import datetime
import json
import os
import boto3
from botocore.exceptions import ClientError


class TrainingPipeline:
    def __init__(self):
        self.model = SentimentModel()
        self.s3_client = boto3.client('s3') if os.getenv('AWS_ACCESS_KEY_ID') else None
        self.bucket_name = os.getenv('S3_BUCKET_NAME')

    async def start_training_job(self, min_samples: int = 1000, retrain_threshold: float = 0.1):
        """Start a new training job"""
        job_id = str(uuid.uuid4())

        async with AsyncSessionLocal() as session:
            # Create training job record
            training_job = TrainingJob(
                job_id=job_id,
                status='pending',
                started_at=datetime.utcnow()
            )
            session.add(training_job)
            await session.commit()

            try:
                # Get training data
                training_data = await self._get_training_data(session, min_samples)

                if len(training_data) < min_samples:
                    raise ValueError(f"Insufficient training data: {len(training_data)} < {min_samples}")

                # Update job status
                training_job.status = 'running'
                training_job.training_samples = len(training_data)
                await session.commit()

                # Train model
                training_results = self.model.train(training_data)

                # Save model
                model_version = f"v_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                model_path = f"models/{model_version}.pth"

                # Save locally first
                local_model_path = f"./models/{model_version}.pth"
                os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
                self.model.save_model(local_model_path)

                # Upload to S3 if available
                if self.s3_client and self.bucket_name:
                    try:
                        self.s3_client.upload_file(local_model_path, self.bucket_name, model_path)
                        model_path = f"s3://{self.bucket_name}/{model_path}"
                    except ClientError as e:
                        print(f"Failed to upload to S3: {e}")
                        model_path = local_model_path
                else:
                    model_path = local_model_path

                # Create model version record
                model_version_record = ModelVersion(
                    version=model_version,
                    model_path=model_path,
                    accuracy=training_results['best_validation_accuracy'],
                    training_samples=training_results['training_samples'],
                    training_config=json.dumps(self.model.config)
                )
                session.add(model_version_record)

                # Update training job
                training_job.status = 'completed'
                training_job.completed_at = datetime.utcnow()
                training_job.validation_accuracy = training_results['best_validation_accuracy']
                training_job.model_version = model_version

                await session.commit()

                # Check if we should deploy this model
                await self._check_and_deploy_model(session, model_version_record)

                return {
                    'job_id': job_id,
                    'model_version': model_version,
                    'validation_accuracy': training_results['best_validation_accuracy'],
                    'training_samples': training_results['training_samples']
                }

            except Exception as e:
                # Update job with error
                training_job.status = 'failed'
                training_job.completed_at = datetime.utcnow()
                training_job.error_message = str(e)
                await session.commit()
                raise e

    async def _get_training_data(self, session, min_samples: int) -> List[Dict]:
        """Get labeled training data from database"""
        query = select(Review).where(
            Review.is_labeled == True,
            Review.sentiment_label.isnot(None)
        )

        result = await session.execute(query)
        reviews = result.scalars().all()

        training_data = []
        for review in reviews:
            training_data.append({
                'text': review.text,
                'label': review.sentiment_label
            })

        return training_data

    async def _check_and_deploy_model(self, session, model_version: ModelVersion):
        """Check if new model should be deployed"""
        # Get current active model
        current_model_query = select(ModelVersion).where(ModelVersion.is_active == True)
        result = await session.execute(current_model_query)
        current_model = result.scalar_one_or_none()

        should_deploy = False

        if not current_model:
            # No active model, deploy this one
            should_deploy = True
        elif model_version.accuracy > current_model.accuracy:
            # New model is better, deploy it
            should_deploy = True

        if should_deploy:
            # Deactivate current model
            if current_model:
                current_model.is_active = False

            # Activate new model
            model_version.is_active = True
            await session.commit()

            print(f"Deployed new model version: {model_version.version}")


# api/routes/training.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from database.database import get_db
from ml.training_pipeline import TrainingPipeline
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/training", tags=["training"])


class TrainingRequest(BaseModel):
    min_samples: int = 1000
    retrain_threshold: float = 0.1


class TrainingResponse(BaseModel):
    job_id: str
    status: str
    message: str


@router.post("/start", response_model=TrainingResponse)
async def start_training(
        request: TrainingRequest,
        background_tasks: BackgroundTasks,
        db: AsyncSession = Depends(get_db)
):
    """Start a new training job"""
    try:
        pipeline = TrainingPipeline()

        # Start training in background
        background_tasks.add_task(
            pipeline.start_training_job,
            min_samples=request.min_samples,
            retrain_threshold=request.retrain_threshold
        )

        return TrainingResponse(
            job_id="background_task",
            status="started",
            message="Training job started in background"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}")
async def get_training_status(job_id: str, db: AsyncSession = Depends(get_db)):
    """Get training job status"""
    from sqlalchemy import select
    from database.models import TrainingJob

    query = select(TrainingJob).where(TrainingJob.job_id == job_id)
    result = await db.execute(query)
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    return {
        'job_id': job.job_id,
        'status': job.status,
        'started_at': job.started_at,
        'completed_at': job.completed_at,
        'training_samples': job.training_samples,
        'validation_accuracy': job.validation_accuracy,
        'error_message': job.error_message,
        'model_version': job.model_version
    }


# api/routes/reviews.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from database.database import get_db
from database.models import Review, ModelVersion
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import json

router = APIRouter(prefix="/reviews", tags=["reviews"])


class ReviewCreate(BaseModel):
    text: str
    source: str
    rating: Optional[int] = None


class ReviewResponse(BaseModel):
    id: int
    text: str
    source: str
    rating: Optional[int]
    sentiment_label: Optional[int]
    confidence_score: Optional[float]
    predicted_sentiment: Optional[str]
    created_at: datetime


class PredictionResponse(BaseModel):
    predicted_sentiment: str
    predicted_class: int
    confidence: float
    probabilities: dict


@router.post("/", response_model=ReviewResponse)
async def create_review(review: ReviewCreate, db: AsyncSession = Depends(get_db)):
    """Create a new review and predict sentiment"""
    # Create review record
    db_review = Review(
        text=review.text,
        source=review.source,
        rating=review.rating,
        created_at=datetime.utcnow()
    )

    db.add(db_review)
    await db.commit()
    await db.refresh(db_review)

    # Predict sentiment using active model
    try:
        prediction = await predict_sentiment(db_review.text, db)

        # Update review with prediction
        db_review.sentiment_label = prediction['predicted_class']
        db_review.confidence_score = prediction['confidence']
        db_review.processed_at = datetime.utcnow()

        await db.commit()

        response = ReviewResponse(
            id=db_review.id,
            text=db_review.text,
            source=db_review.source,
            rating=db_review.rating,
            sentiment_label=db_review.sentiment_label,
            confidence_score=db_review.confidence_score,
            predicted_sentiment=prediction['predicted_sentiment'],
            created_at=db_review.created_at
        )

        return response

    except Exception as e:
        # Return review without prediction if model fails
        return ReviewResponse(
            id=db_review.id,
            text=db_review.text,
            source=db_review.source,
            rating=db_review.rating,
            sentiment_label=None,
            confidence_score=None,
            predicted_sentiment=None,
            created_at=db_review.created_at
        )


@router.get("/", response_model=List[ReviewResponse])
async def get_reviews(
        skip: int = 0,
        limit: int = 100,
        source: Optional[str] = None,
        sentiment_label: Optional[int] = None,
        db: AsyncSession = Depends(get_db)
):
    """Get reviews with filters"""
    query = select(Review)

    if source:
        query = query.where(Review.source == source)
    if sentiment_label is not None:
        query = query.where(Review.sentiment_label == sentiment_label)

    query = query.offset(skip).limit(limit).order_by(Review.created_at.desc())

    result = await db.execute(query)
    reviews = result.scalars().all()

    label_names = ['Negative', 'Neutral', 'Positive', 'Constructive']

    return [
        ReviewResponse(
            id=review.id,
            text=review.text,
            source=review.source,
            rating=review.rating,
            sentiment_label=review.sentiment_label,
            confidence_score=review.confidence_score,
            predicted_sentiment=label_names[review.sentiment_label] if review.sentiment_label is not None else None,
            created_at=review.created_at
        )
        for review in reviews
    ]


@router.post("/{review_id}/label")
async def label_review(
        review_id: int,
        sentiment_label: int,
        db: AsyncSession = Depends(get_db)
):
    """Manually label a review for training"""
    if sentiment_label not in [0, 1, 2, 3]:
        raise HTTPException(status_code=400, detail="Invalid sentiment label")

    query = select(Review).where(Review.id == review_id)
    result = await db.execute(query)
    review = result.scalar_one_or_none()

    if not review:
        raise HTTPException(status_code=404, detail="Review not found")

    review.sentiment_label = sentiment_label
    review.is_labeled = True
    review.is_training_data = True

    await db.commit()

    return {"message": "Review labeled successfully"}


@router.post("/predict", response_model=PredictionResponse)
async def predict_review_sentiment(text: str, db: AsyncSession = Depends(get_db)):
    """Predict sentiment for a text"""
    prediction = await predict_sentiment(text, db)
    return PredictionResponse(**prediction)


async def predict_sentiment(text: str, db: AsyncSession):
    """Helper function to predict sentiment using active model"""
    # Get active model
    query = select(ModelVersion).where(ModelVersion.is_active == True)
    result = await db.execute(query)
    active_model = result.scalar_one_or_none()

    if not active_model:
        raise HTTPException(status_code=503, detail="No active model available")

    # Load and use model
    from ml.model import SentimentModel
    model = SentimentModel()

    # Handle both local and S3 paths
    model_path = active_model.model_path
    if model_path.startswith('s3://'):
        # Download from S3 if needed
        import boto3
        s3_client = boto3.client('s3')
        bucket = model_path.split('/')[2]
        key = '/'.join(model_path.split('/')[3:])
        local_path = f"./temp_models/{active_model.version}.pth"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client.download_file(bucket, key, local_path)
        model_path = local_path

    model.load_model(model_path)
    prediction = model.predict(text)

    return prediction


# api/routes/analytics.py
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from database.database import get_db
from database.models import Review, ModelVersion, TrainingJob
from datetime import datetime, timedelta
from typing import Optional

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/dashboard")
async def get_dashboard_stats(
        days: int = 30,
        db: AsyncSession = Depends(get_db)
):
    """Get dashboard statistics"""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    # Total reviews
    total_reviews_query = select(func.count(Review.id))
    total_reviews_result = await db.execute(total_reviews_query)
    total_reviews = total_reviews_result.scalar()

    # Recent reviews
    recent_reviews_query = select(func.count(Review.id)).where(
        Review.created_at >= start_date
    )
    recent_reviews_result = await db.execute(recent_reviews_query)
    recent_reviews = recent_reviews_result.scalar()

    # Sentiment distribution
    sentiment_query = select(
        Review.sentiment_label,
        func.count(Review.id).label('count')
    ).where(
        Review.sentiment_label.isnot(None),
        Review.created_at >= start_date
    ).group_by(Review.sentiment_label)

    sentiment_result = await db.execute(sentiment_query)
    sentiment_distribution = {}
    label_names = ['Negative', 'Neutral', 'Positive', 'Constructive']

    for label, count in sentiment_result:
        sentiment_distribution[label_names[label]] = count

    # Model performance
    active_model_query = select(ModelVersion).where(ModelVersion.is_active == True)
    active_model_result = await db.execute(active_model_query)
    active_model = active_model_result.scalar_one_or_none()

    # Recent training jobs
    training_jobs_query = select(TrainingJob).order_by(
        TrainingJob.started_at.desc()
    ).limit(5)
    training_jobs_result = await db.execute(training_jobs_query)
    recent_training_jobs = training_jobs_result.scalars().all()

    return {
        'total_reviews': total_reviews,
        'recent_reviews': recent_reviews,
        'sentiment_distribution': sentiment_distribution,
        'active_model': {
            'version': active_model.version if active_model else None,
            'accuracy': active_model.accuracy if active_model else None,
            'training_samples': active_model.training_samples if active_model else None,
            'created_at': active_model.created_at if active_model else None
        },
        'recent_training_jobs': [
            {
                'job_id': job.job_id,
                'status': job.status,
                'started_at': job.started_at,
                'validation_accuracy': job.validation_accuracy,
                'model_version': job.model_version
            }
            for job in recent_training_jobs
        ]
    }


@router.get("/constructive-feedback")
async def get_constructive_feedback(
        days: int = 7,
        skip: int = 0,
        limit: int = 50,
        db: AsyncSession = Depends(get_db)
):
    """Get constructive feedback for business insights"""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    query = select(Review).where(
        Review.sentiment_label == 3,  # Constructive
        Review.created_at >= start_date
    ).order_by(
        Review.confidence_score.desc()
    ).offset(skip).limit(limit)

    result = await db.execute(query)
    constructive_reviews = result.scalars().all()

    # Extract insights for each review
    insights = []
    for review in constructive_reviews:
        # Simple keyword extraction for insights
        insight = extract_business_insights(review.text)
        insights.append({
            'id': review.id,
            'text': review.text,
            'source': review.source,
            'confidence': review.confidence_score,
            'created_at': review.created_at,
            'insights': insight
        })

    return {
        'constructive_feedback': insights,
        'total_count': len(insights)
    }


def extract_business_insights(text: str) -> dict:
    """Extract business insights from constructive feedback"""
    text_lower = text.lower()

    # Define categories and keywords
    categories = {
        'service': ['service', 'staff', 'employee', 'wait', 'slow', 'fast', 'friendly'],
        'product': ['product', 'quality', 'food', 'taste', 'fresh', 'stale'],
        'facility': ['location', 'parking', 'space', 'clean', 'dirty', 'lighting', 'temperature'],
        'process': ['checkout', 'delivery', 'ordering', 'payment', 'system', 'app', 'website'],
        'pricing': ['price', 'expensive', 'cheap', 'value', 'cost', 'worth']
    }

    # Extract improvement suggestions
    improvement_phrases = [
        'could', 'should', 'would', 'better', 'improve', 'consider', 'suggest', 'maybe', 'perhaps'
    ]

    detected_categories = []
    suggestions = []

    for category, keywords in categories.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_categories.append(category)

    # Look for improvement suggestions
    sentences = text.split('.')
    for sentence in sentences:
        if any(phrase in sentence.lower() for phrase in improvement_phrases):
            suggestions.append(sentence.strip())

    return {
        'categories': detected_categories,
        'suggestions': suggestions[:3],  # Top 3 suggestions
        'actionable': len(suggestions) > 0
    }


# scheduler/tasks.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from ml.training_pipeline import TrainingPipeline
import asyncio
import logging

logger = logging.getLogger(__name__)


class MLScheduler:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.training_pipeline = TrainingPipeline()

    def start(self):
        """Start the scheduler"""
        # Schedule daily model retraining at 2 AM
        self.scheduler.add_job(
            func=self.daily_retrain,
            trigger=CronTrigger(hour=2, minute=0),
            id='daily_retrain',
            name='Daily Model Retraining',
            replace_existing=True
        )

        # Schedule weekly model evaluation at Sunday 1 AM
        self.scheduler.add_job(
            func=self.weekly_evaluation,
            trigger=CronTrigger(day_of_week=6, hour=1, minute=0),
            id='weekly_evaluation',
            name='Weekly Model Evaluation',
            replace_existing=True
        )

        self.scheduler.start()
        logger.info("ML Scheduler started")

    async def daily_retrain(self):
        """Daily retraining task"""
        try:
            logger.info("Starting daily model retraining")
            result = await self.training_pipeline.start_training_job(
                min_samples=500,  # Lower threshold for daily training
                retrain_threshold=0.05
            )
            logger.info(f"Daily retraining completed: {result}")
        except Exception as e:
            logger.error(f"Daily retraining failed: {e}")

    async def weekly_evaluation(self):
        """Weekly model evaluation task"""
        try:
            logger.info("Starting weekly model evaluation")
            # Add comprehensive evaluation logic here
            await self.evaluate_model_performance()
            logger.info("Weekly evaluation completed")
        except Exception as e:
            logger.error(f"Weekly evaluation failed: {e}")

    async def evaluate_model_performance(self):
        """Evaluate current model performance"""
        from database.database import AsyncSessionLocal
        from database.models import Review, ModelVersion
        from sqlalchemy import select
        from datetime import datetime, timedelta

        async with AsyncSessionLocal() as session:
            # Get recent reviews for evaluation
            week_ago = datetime.utcnow() - timedelta(days=7)
            query = select(Review).where(
                Review.created_at >= week_ago,
                Review.is_labeled == True,
                Review.sentiment_label.isnot(None)
            )

            result = await session.execute(query)
            recent_reviews = result.scalars().all()

            if len(recent_reviews) < 50:
                logger.warning("Insufficient labeled data for evaluation")
                return

            # Load active model and evaluate
            active_model_query = select(ModelVersion).where(ModelVersion.is_active == True)
            model_result = await session.execute(active_model_query)
            active_model = model_result.scalar_one_or_none()

            if not active_model:
                logger.warning("No active model found")
                return

            # Calculate accuracy on recent data
            correct_predictions = 0
            total_predictions = 0

            from ml.model import SentimentModel
            model = SentimentModel()
            model.load_model(active_model.model_path)

            for review in recent_reviews:
                try:
                    prediction = model.predict(review.text)
                    if prediction['predicted_class'] == review.sentiment_label:
                        correct_predictions += 1
                    total_predictions += 1
                except Exception as e:
                    logger.error(f"Prediction failed for review {review.id}: {e}")

            if total_predictions > 0:
                accuracy = correct_predictions / total_predictions
                logger.info(f"Current model accuracy on recent data: {accuracy:.3f}")

                # If accuracy drops significantly, trigger retraining
                if accuracy < 0.7:  # Threshold for retraining
                    logger.warning(f"Model accuracy too low ({accuracy:.3f}), triggering retraining")
                    await self.training_pipeline.start_training_job(min_samples=1000)

    def stop(self):
        """Stop the scheduler"""
        self.scheduler.shutdown()
        logger.info("ML Scheduler stopped")


# main.py - FastAPI application
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from api.routes import training, reviews, analytics
from scheduler.tasks import MLScheduler
from database.database import sync_engine
from database.models import Base
import uvicorn
import logging
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global scheduler instance
ml_scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global ml_scheduler

    # Startup
    logger.info("Starting ML Pipeline API")

    # Create database tables
    Base.metadata.create_all(bind=sync_engine)

    # Start scheduler
    ml_scheduler = MLScheduler()
    ml_scheduler.start()

    yield

    # Shutdown
    if ml_scheduler:
        ml_scheduler.stop()

    logger.info("ML Pipeline API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Sentiment Analysis ML Pipeline",
    description="FastAPI-based ML pipeline for sentiment analysis with automated training",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(training.router)
app.include_router(reviews.router)
app.include_router(analytics.router)


@app.get("/")
async def root():
    return {
        "message": "Sentiment Analysis ML Pipeline API",
        "version": "1.0.0",
        "endpoints": {
            "reviews": "/reviews",
            "training": "/training",
            "analytics": "/analytics"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from database.database import AsyncSessionLocal

    try:
        # Test database connection
        async with AsyncSessionLocal() as session:
            await session.execute("SELECT 1")

        return {
            "status": "healthy",
            "database": "connected",
            "scheduler": "running" if ml_scheduler else "stopped"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# docker-compose.yml
"""
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: sentiment_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  mlflow:
    image: python:3.9
    command: bash -c "pip install mlflow psycopg2-binary && mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri postgresql://postgres:password@postgres:5432/sentiment_db --default-artifact-root ./mlruns"
    ports:
      - "5000:5000"
    depends_on:
      - postgres
    volumes:
      - mlflow_data:/app/mlruns

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/sentiment_db
      - REDIS_URL=redis://redis:6379
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - postgres
      - redis
      - mlflow
    volumes:
      - ./models:/app/models
      - ./temp_models:/app/temp_models

volumes:
  postgres_data:
  mlflow_data:
"""

# Dockerfile
"""
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for models
RUN mkdir -p models temp_models

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
"""

# alembic/env.py (for database migrations)
"""
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
from database.models import Base
import os

# This is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata
target_metadata = Base.metadata

# Set database URL from environment
config.set_main_option('sqlalchemy.url', os.getenv('DATABASE_URL'))

def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
"""

# Usage Example Script
"""
# example_usage.py
import asyncio
import aiohttp
import json

async def example_usage():
    base_url = "http://localhost:8000"

    # 1. Create some reviews
    reviews = [
        {"text": "Great service and excellent food quality!", "source": "google", "rating": 5},
        {"text": "Terrible experience, worst restaurant ever.", "source": "yelp", "rating": 1},
        {"text": "Good food but service was slow, maybe hire more staff.", "source": "internal", "rating": 3},
        {"text": "Nice atmosphere but parking is limited, consider valet service.", "source": "google", "rating": 4}
    ]

    async with aiohttp.ClientSession() as session:
        # Create reviews
        for review in reviews:
            async with session.post(f"{base_url}/reviews/", json=review) as response:
                result = await response.json()
                print(f"Created review: {result['predicted_sentiment']} (confidence: {result['confidence_score']:.3f})")

        # Label some reviews for training
        await session.post(f"{base_url}/reviews/1/label", params={"sentiment_label": 2})
        await session.post(f"{base_url}/reviews/2/label", params={"sentiment_label": 0})
        await session.post(f"{base_url}/reviews/3/label", params={"sentiment_label": 3})
        await session.post(f"{base_url}/reviews/4/label", params={"sentiment_label": 3})

        # Start training
        async with session.post(f"{base_url}/training/start", json={"min_samples": 4}) as response:
            training_result = await response.json()
            print(f"Training started: {training_result}")

        # Get analytics
        async with session.get(f"{base_url}/analytics/dashboard") as response:
            dashboard = await response.json()
            print(f"Dashboard stats: {json.dumps(dashboard, indent=2)}")

        # Get constructive feedback
        async with session.get(f"{base_url}/analytics/constructive-feedback") as response:
            feedback = await response.json()
            print(f"Constructive feedback: {json.dumps(feedback, indent=2)}")

if __name__ == "__main__":
    asyncio.run(example_usage())
"""