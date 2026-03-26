# requirements.txt: fastapi, uvicorn, mlx-lm, mlx, psutil, pydantic

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

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mlx-engine-local")

app = FastAPI(title="mlxEngine (Local): Robust Model Server")
security = HTTPBearer()

# --- Configuration & Constants ---
DEFAULT_CONFIG = {
    "model_root": "../Warehouse/mlx_models",
    "port": 4474,
    "api_key": "hnxj-m3max-key",
    "memory_ceiling_gb": 80.0,
    "last_model": None
}
CONFIG_FILE = "config.json"
STATUS_FILE = "engine-status.md"

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.current_model_name = None
        self.is_vlm = False
        self.status = "idle"
        self.config = DEFAULT_CONFIG.copy()
        self.load_config()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    user_config = json.load(f)
                    self.config.update(user_config)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")

    def save_config(self):
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=4)

    def log_status(self, message: str):
        with open(STATUS_FILE, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

    def unload_model(self):
        if self.model is not None:
            logger.info(f"Unloading model: {self.current_model_name}")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.current_model_name = None
            try:
                mx.clear_cache()
            except AttributeError:
                mx.metal.clear_cache()
            gc.collect()
            logger.info("Model unloaded and cache cleared.")

    def get_vram_usage_gb(self):
        try:
            return mx.get_active_memory() / (1024**3)
        except AttributeError:
            return mx.metal.get_active_memory() / (1024**3)

    async def load_model(self, model_name: str):
        model_root = self.config.get("model_root", "../Warehouse/mlx_models")
        model_path = os.path.join(model_root, model_name)
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found in {model_root}")

        self.status = "loading"
        self.unload_model()
        
        logger.info(f"Loading model: {model_name}...")
        try:
            # Run load in a thread to not block the event loop
            loop = asyncio.get_event_loop()
            self.model, self.tokenizer = await loop.run_in_executor(None, lambda: mlx_load(model_path))
            self.current_model_name = model_name
            self.status = "ready"
            self.config["last_model"] = model_name
            self.save_config()
            self.log_status(f"Loaded {model_name}")
            logger.info(f"Successfully loaded {model_name}")
        except Exception as e:
            self.status = "error"
            logger.error(f"Failed to load model {model_name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

manager = ModelManager()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != manager.config.get("api_key"):
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return credentials.credentials

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False

@app.get("/status")
async def get_status():
    return {
        "status": manager.status,
        "current_model": manager.current_model_name,
        "vram_usage_gb": f"{manager.get_vram_usage_gb():.2f}",
        "memory_ceiling_gb": manager.config.get("memory_ceiling_gb")
    }

@app.get("/models")
async def list_models(token: str = Depends(verify_token)):
    model_root = manager.config.get("model_root")
    if not os.path.exists(model_root): return []
    return [d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))]

@app.post("/load_model")
async def load_model_endpoint(request: Dict[str, str], token: str = Depends(verify_token)):
    model_name = request.get("model")
    await manager.load_model(model_name)
    return {"status": "success", "model": model_name}

import mlx_vlm
from mlx_vlm.utils import load as vlm_load
from mlx_vlm.generate import generate as vlm_generate

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.current_model_name = None
        self.is_vlm = False
        self.status = "idle"
        self.config = DEFAULT_CONFIG.copy()
        self.load_config()

    # ... (unload_model unchanged)
    def unload_model(self):
        if self.model is not None:
            logger.info(f"Unloading model: {self.current_model_name}")
            del self.model
            del self.tokenizer
            if self.processor: del self.processor
            self.model = None
            self.tokenizer = None
            self.processor = None
            self.current_model_name = None
            self.is_vlm = False
            try:
                mx.clear_cache()
            except AttributeError:
                mx.metal.clear_cache()
            gc.collect()

    async def load_model(self, model_name: str):
        model_root = self.config.get("model_root", "../Warehouse/mlx_models")
        model_path = os.path.join(model_root, model_name)
        
        self.unload_model()
        logger.info(f"Loading model: {model_name}...")
        
        # Determine if it's a VLM (Gemma 3 or contains 'vl')
        self.is_vlm = "gemma-3" in model_name.lower() or "vl" in model_name.lower()
        
        try:
            loop = asyncio.get_event_loop()
            if self.is_vlm:
                self.model, self.processor = await loop.run_in_executor(None, lambda: vlm_load(model_path))
                self.tokenizer = self.processor.tokenizer
            else:
                self.model, self.tokenizer = await loop.run_in_executor(None, lambda: mlx_load(model_path))
            
            self.current_model_name = model_name
            self.status = "ready"
            self.config["last_model"] = model_name
            self.save_config()
            logger.info(f"Successfully loaded {model_name} (VLM={self.is_vlm})")
        except Exception as e:
            self.status = "error"
            logger.error(f"Failed to load {model_name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, token: str = Depends(verify_token)):
    if manager.status != "ready" or manager.current_model_name != request.model:
         await manager.load_model(request.model)

    manager.status = "thinking"
    
    images = []
    processed_messages = []
    for msg in request.messages:
        if isinstance(msg.content, list):
            msg_text = ""
            for item in msg.content:
                if item.get("type") == "text":
                    msg_text += item.get("text", "")
                elif item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:image"):
                        base64_data = url.split(",")[1]
                        img_data = base64.b64decode(base64_data)
                        images.append(Image.open(BytesIO(img_data)))
            processed_messages.append({"role": msg.role, "content": msg_text})
        else:
            processed_messages.append(msg.model_dump())

    def generate_response():
        if manager.is_vlm:
            # Gemma 3 template fix for VLM
            prompt = manager.tokenizer.apply_chat_template(processed_messages, tokenize=False, add_generation_prompt=True)
            response = vlm_generate(manager.model, manager.processor, prompt, images, max_tokens=request.max_tokens, verbose=False)
        else:
            prompt = manager.tokenizer.apply_chat_template(processed_messages, tokenize=False, add_generation_prompt=True)
            sampler = make_sampler(request.temperature, request.top_p)
            response = generate(manager.model, manager.tokenizer, prompt, max_tokens=request.max_tokens, sampler=sampler, verbose=False)
        
        manager.status = "ready"
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 0}
        }

    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming not yet implemented")
    
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, generate_response)
        return result
    except Exception as e:
        manager.status = "ready"
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    last_model = manager.config.get("last_model")
    if last_model:
        try:
            logger.info(f"Auto-loading last model: {last_model}")
            await manager.load_model(last_model)
        except Exception as e:
            logger.warning(f"Could not auto-load {last_model}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=manager.config.get("port"))
