from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import query, health
from ..config.settings import settings
from ..utils.logging_config import logger


def create_app():
    app = FastAPI(
        title="RAG Chatbot Backend for Physical AI & Humanoid Robotics Textbook",
        description="API for question answering based on Physical AI & Humanoid Robotics textbook content",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(query.router)
    app.include_router(health.router)

    return app


app = create_app()


@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "RAG Chatbot Backend for Physical AI & Humanoid Robotics Textbook"}


@app.on_event("startup")
async def startup_event():
    logger.info("Application startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown initiated")