from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime
import base64

# Image generation import
from emergentintegrations.llm.openai.image_generation import OpenAIImageGeneration

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Initialize image generator
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')
image_gen = OpenAIImageGeneration(api_key=EMERGENT_LLM_KEY)

# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

class ImageGenerationRequest(BaseModel):
    prompt: str
    style: Optional[str] = "default"

class ImageGenerationResponse(BaseModel):
    image_base64: str
    prompt: str
    style: str
    generated_at: datetime

class GeneratedImage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    image_base64: str
    prompt: str
    style: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)

# Style modifiers for prompts
STYLE_MODIFIERS = {
    "cartoon": "in a fun cartoon style, vibrant colors, playful",
    "neon": "with neon glow effects, cyberpunk style, bright fluorescent colors",
    "retro": "vintage 80s style, retro aesthetics, nostalgic feel",
    "watercolor": "painted in watercolor style, soft brushstrokes, artistic",
    "fantasy": "fantasy art style, magical and dreamy, ethereal"
}

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "AI Image Creator API Ready!"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

@api_router.post("/generate-image", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest):
    try:
        # Enhance prompt with style
        enhanced_prompt = request.prompt
        if request.style and request.style != "default" and request.style in STYLE_MODIFIERS:
            enhanced_prompt = f"{request.prompt}, {STYLE_MODIFIERS[request.style]}"
        
        # Generate image using emergentintegrations
        images = await image_gen.generate_images(
            prompt=enhanced_prompt,
            model="gpt-image-1",
            number_of_images=1
        )
        
        if images and len(images) > 0:
            # Convert image to base64
            image_base64 = base64.b64encode(images[0]).decode('utf-8')
            
            # Save to database
            generated_image = GeneratedImage(
                image_base64=image_base64,
                prompt=request.prompt,
                style=request.style or "default"
            )
            
            await db.generated_images.insert_one(generated_image.dict())
            
            return ImageGenerationResponse(
                image_base64=image_base64,
                prompt=request.prompt,
                style=request.style or "default",
                generated_at=generated_image.generated_at
            )
        else:
            raise HTTPException(status_code=500, detail="No image was generated")
            
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate image: {str(e)}")

@api_router.get("/gallery", response_model=List[GeneratedImage])
async def get_gallery(limit: int = 20):
    """Get recent generated images for gallery"""
    try:
        images = await db.generated_images.find().sort("generated_at", -1).limit(limit).to_list(limit)
        return [GeneratedImage(**image) for image in images]
    except Exception as e:
        logger.error(f"Error fetching gallery: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch gallery")

@api_router.get("/styles")
async def get_available_styles():
    """Get available style options"""
    return {
        "styles": [
            {"key": "default", "name": "Default", "description": "Natural style"},
            {"key": "cartoon", "name": "Cartoon", "description": "Fun cartoon style"},
            {"key": "neon", "name": "Neon", "description": "Cyberpunk neon glow"},
            {"key": "retro", "name": "Retro", "description": "Vintage 80s style"},
            {"key": "watercolor", "name": "Watercolor", "description": "Soft watercolor painting"},
            {"key": "fantasy", "name": "Fantasy", "description": "Magical fantasy art"}
        ]
    }

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()