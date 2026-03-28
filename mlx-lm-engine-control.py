import os
import gc
import json
import time
import psutil
import logging
import asyncio
import base64
from io import BytesIO
from PIL import Image
from typing import List, Optional, Union, Dict, Any
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

import mlx.core as mx
from mlx_lm.utils import load as mlx_load
from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler

import mlx_vlm
from mlx_vlm.utils import load as vlm_load
from mlx_vlm.generate import generate as vlm_generate

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mlx-engine-local")

app = FastAPI(title="mlxEngine (Local): Hybrid Multimodal Server")
security = HTTPBearer()

DEFAULT_CONFIG = {
    "model_root": "../Warehouse/mlx_models",
    "port": 4474,
    "api_key": "hnxj-m3max-key",
    "memory_ceiling_gb": 80.0,
    "last_model": None
}
CONFIG_FILE = "config.json"

class LoadedModel:
    def __init__(self, model, tokenizer, processor=None, is_vlm=False):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.is_vlm = is_vlm

class ModelManager:
    def __init__(self):
        self.models: Dict[str, LoadedModel] = {}
        self.status = "idle"
        self.config = DEFAULT_CONFIG.copy()
        self.load_config()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    self.config.update(json.load(f))
            except Exception as e:
                logger.error(f"Failed to load config: {e}")

    def save_config(self):
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=4)

    def unload_all(self):
        logger.info("Unloading all models")
        self.models.clear()
        mx.clear_cache()
        gc.collect()

    def unload_model(self, model_name: str):
        if model_name in self.models:
            logger.info(f"Unloading model: {model_name}")
            del self.models[model_name]
            mx.clear_cache()
            gc.collect()

    async def load_model(self, model_name: str):
        if model_name in self.models:
            return

        model_root = self.config.get("model_root", "../Warehouse/mlx_models")
        model_path = os.path.join(model_root, model_name)
        if not os.path.exists(model_path):
            # Try absolute path
            if os.path.exists(model_name):
                model_path = model_name
            else:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        # Check VRAM ceiling before loading (rough estimate: check current usage)
        current_usage = self.get_vram_usage_gb()
        if current_usage > self.config.get("memory_ceiling_gb", 80.0):
            logger.warning(f"VRAM usage {current_usage:.2f}GB exceeds ceiling. Unloading all before load.")
            self.unload_all()

        self.status = "loading"
        is_vlm = "vl" in model_name.lower() or "gemma-3" in model_name.lower()
        
        try:
            loop = asyncio.get_event_loop()
            if is_vlm:
                model, processor = await loop.run_in_executor(None, lambda: vlm_load(model_path))
                self.models[model_name] = LoadedModel(model, processor.tokenizer, processor, True)
            else:
                model, tokenizer = await loop.run_in_executor(None, lambda: mlx_load(model_path))
                self.models[model_name] = LoadedModel(model, tokenizer, is_vlm=False)
            
            self.status = "ready"
            self.config["last_model"] = model_name
            self.save_config()
            logger.info(f"Loaded {model_name} (VLM={is_vlm})")
        except Exception as e:
            self.status = "error"
            logger.error(f"Load error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def get_vram_usage_gb(self):
        # mx.metal.get_active_memory() is deprecated in newer MLX, but using it as in original script
        try:
            return mx.metal.get_active_memory() / (1024**3)
        except:
            return mx.get_active_memory() / (1024**3)

manager = ModelManager()

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    stream: Optional[bool] = False

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != manager.config.get("api_key"):
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return credentials.credentials

@app.get("/status")
async def get_status():
    return {
        "status": manager.status, 
        "loaded_models": list(manager.models.keys()), 
        "vram_usage_gb": f"{manager.get_vram_usage_gb():.2f}"
    }

class LoadModelRequest(BaseModel):
    model: str

@app.post("/load_model")
async def load_model(request: LoadModelRequest, token: str = Depends(verify_token)):
    await manager.load_model(request.model)
    return {"status": "success", "message": f"Loaded {request.model}"}

@app.post("/unload_all")
async def unload_all(token: str = Depends(verify_token)):
    manager.unload_all()
    return {"status": "success", "message": "All models unloaded"}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, token: str = Depends(verify_token)):
    if request.model not in manager.models:
        await manager.load_model(request.model)

    lm = manager.models[request.model]
    manager.status = "thinking"
    
    if lm.is_vlm:
        images = []
        vlm_messages = []
        for msg in request.messages:
            content_list = []
            if isinstance(msg.content, list):
                for item in msg.content:
                    if item.get("type") == "text":
                        content_list.append(item)
                    elif item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:image"):
                            base64_data = url.split(",")[1]
                            img = Image.open(BytesIO(base64.b64decode(base64_data)))
                            images.append(img)
                            content_list.append({"type": "image"})
            else:
                content_list.append({"type": "text", "text": msg.content})
            vlm_messages.append({"role": msg.role, "content": content_list})
        
        try:
            loop = asyncio.get_event_loop()
            prompt = lm.processor.apply_chat_template(vlm_messages, add_generation_prompt=True)
            response = await loop.run_in_executor(None, lambda: vlm_generate(lm.model, lm.processor, prompt, images, max_tokens=request.max_tokens, verbose=False))
        except Exception as e:
            manager.status = "ready"
            raise HTTPException(status_code=500, detail=f"VLM Gen Error: {e}")
    else:
        # Standard Text Generation
        prompt = lm.tokenizer.apply_chat_template([m.model_dump() for m in request.messages], tokenize=False, add_generation_prompt=True)
        loop = asyncio.get_event_loop()
        sampler = make_sampler(request.temperature)
        response = await loop.run_in_executor(None, lambda: generate(lm.model, lm.tokenizer, prompt, max_tokens=request.max_tokens, sampler=sampler))

    manager.status = "ready"
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 0}
    }

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="mlxEngine (Local): Hybrid Multimodal Server")
    parser.add_argument("--port", type=int, help="Port to run the server on (overrides config.json)")
    args = parser.parse_args()
    
    port = args.port if args.port else manager.config.get("port", 4474)
    logger.info(f"🚀 Starting mlxEngine on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
