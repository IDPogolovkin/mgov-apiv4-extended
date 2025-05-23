import os
import json
import logging
import asyncio
import uuid
import hashlib
from datetime import datetime
from typing import List, Dict
import requests
import httpx
import redis.asyncio as redis
import torch
import uvicorn
import asyncpg
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pymilvus import connections, Collection, utility
import secrets

from FlagEmbedding import BGEM3FlagModel 

# Load environment variables
load_dotenv()

# Set up the HTTPBearer security scheme
security = HTTPBearer()

# Bearer authentication dependency
async def verify_bearer_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing credentials")
    if not secrets.compare_digest(credentials.credentials, API_BEARER_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from config import (
    OPENAI_API_KEY,
    MILVUS_HOST,
    MILVUS_PORT,
    REDIS_URL,
    DB_CONFIG,
    REDIS_UNAVLB_MSSG, 
    CHAT_HISTORY_TTL,
    API_BEARER_TOKEN,
    SYSTEM_PROMPT,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize service status dictionary
    app.state.services_status = {
        "redis": False,
        "zilliz": False,
        "database": False,
        "embeddings": False
    }

    redis_url = os.getenv("REDIS_URL", REDIS_URL)
    # decode_responses=True returns strings instead of bytes
    app.state.redis = redis.from_url(
        redis_url,
        encoding="utf-8",
        decode_responses=True
    )
    try:
        # ping() returns True or raises
        if await app.state.redis.ping():
            logger.info("Redis connection established")
            app.state.services_status["redis"] = True
        else:
            logger.warning("Redis ping failed")
            app.state.redis = None
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        app.state.redis = None

    # Attempt to load Zilliz collection
    try:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
        collection_name = "egov_general_2_ru"  # Make sure it matches your local collection
        collection = Collection(collection_name)
        collection.load()
        
        app.state.collection = collection
        app.state.services_status["milvus"] = True
        logger.info(f"Connected and loaded collection '{collection_name}'")
    except Exception as e:
        logger.error(f"Milvus connection failed: {e}")
        app.state.collection = None
        app.state.services_status["milvus"] = False
        
    # Create DB connection pool
    try:
        app.state.db_pool = await asyncpg.create_pool(**DB_CONFIG)
        # Test connection
        async with app.state.db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        logger.info("Database connection established")
        app.state.services_status["database"] = True
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        app.state.db_pool = None

    # Initialize the embedding model
    app.state.embeddings_model = None
    try:
        if torch.cuda.is_available():
            # FP16 + GPU
            model = BGEM3FlagModel(
                'BAAI/bge-m3',
                use_fp16=True,
                device="cuda"
            )
            logger.info("CUDA available ➞ loaded BGE-M3 on GPU with FP16")
        else:
            # FP32 + CPU
            model = BGEM3FlagModel(
                'BAAI/bge-m3',
                use_fp16=False
            )
            logger.info("CUDA not available ➞ loaded BGE-M3 on CPU with FP32")

        app.state.embeddings_model = model
        app.state.services_status["embeddings"] = True
        logger.info("Embeddings model initialized successfully")

    except Exception as e:
        logger.error(f"Embedding model initialization error: {e}")

    # Initialize httpx client for OpenAI API
    app.state.httpx_client = httpx.AsyncClient(
        timeout=httpx.Timeout(60.0),
        http2=True,
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=80),
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
    )

    # Register background tasks
    app.state.background_tasks = set()
    
    # Start expired thread cleanup task
    background_task = asyncio.create_task(clean_expired_threads(app))
    app.state.background_tasks.add(background_task)
    background_task.add_done_callback(app.state.background_tasks.discard)

    logger.info("Application started, resources initialized")
    yield

    # Cancel background tasks
    for task in app.state.background_tasks:
        task.cancel()

    # Cleanup on shutdown
    if app.state.redis:
        # close the connection pool
        await app.state.redis.close()
        # and wait for it to shut down
        await app.state.redis.wait_closed()
    if hasattr(app.state, 'db_pool') and app.state.db_pool is not None:
        await app.state.db_pool.close()
    await app.state.httpx_client.aclose()
    try:
        if hasattr(app.state, 'collection') and app.state.collection:
            app.state.collection.release()
            logger.info("Milvus collection released.")
    except Exception as e:
        logger.error(f"Failed to release Milvus collection: {e}")
    connections.disconnect(alias="default")
    logger.info("Application shutting down, resources released")

# Attach the global Bearer authentication dependency to all endpoints by adding it to the app.
app = FastAPI(lifespan=lifespan, dependencies=[Depends(verify_bearer_token)])

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET","POST"],
    allow_headers=["*"],
)

# -----------------------
# Utility and Dependencies
# -----------------------
class EgovAiRequest(BaseModel):
    user_question: str = Field(..., description="User question about eGov services")
    cache_response: bool = Field(False, description="Whether to cache the response")

async def get_db_conn(request: Request):
    async with request.app.state.db_pool.acquire() as conn:
        yield conn

async def get_redis(request: Request):
    return request.app.state.redis

def get_model(request: Request):
    return request.app.state.embeddings_model

def get_collection(request: Request):
    return request.app.state.collection

def get_httpx_client(request: Request):
    return request.app.state.httpx_client

def generate_cache_key(question: str) -> str:
    hash_digest = hashlib.sha256(question.encode('utf-8')).hexdigest()
    return f"egov_ai:{hash_digest}"

async def get_embedding_async(query: str, model) -> List[float]:
    """
    Asynchronously generate an embedding for the given query using the provided FlagEmbedding model.
    Uses torch.no_grad() for inference and the best practice parameters: batch_size and max_length.
    """
    def _generate_embedding():
        with torch.no_grad():
            output = model.encode(
                [query],
                batch_size=12,
                max_length=8192,
            )
            # The flag embedding model returns a dict with key 'dense_vecs'
            return output['dense_vecs'][0]
    return await asyncio.to_thread(_generate_embedding)

# Background task to clean expired threads
async def clean_expired_threads(app):
    """Periodically clean up expired chat history keys."""
    while True:
        try:
            if app.state.redis:
                cursor = "0"
                deleted_count = 0
                # Use pipeline to group calls, reducing network overhead
                while True:
                    cursor, keys = await app.state.redis.scan(cursor=cursor, match="chat_history:*", count=100)
                    if keys:
                        # Create a pipeline for multiple TTL calls
                        pipe = app.state.redis.pipeline()
                        for key in keys:
                            pipe.ttl(key)
                        ttls = await pipe.execute()

                        # Create another pipeline for deletions if needed
                        pipe = app.state.redis.pipeline()
                        for key, ttl in zip(keys, ttls):
                            if ttl < 0:
                                pipe.delete(key)
                                deleted_count += 1
                        await pipe.execute()
                    if cursor == "0":
                        break

                if deleted_count > 0:
                    # Log a summary rather than per-key logs
                    logger.info(f"Cleaned up {deleted_count} expired thread(s).")
            # Run every 10 minutes (adjust as needed)
            await asyncio.sleep(600)
        except asyncio.CancelledError:
            logger.info("Thread cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in thread cleanup task: {e}")
            # Pause briefly on errors to avoid rapid error looping
            await asyncio.sleep(60)

# -----------------------
# Search Functions
# -----------------------
def get_collections(request: Request):
    return request.app.state.collection

async def search_milvus(
    query: str,
    embedding: List[float],
    collection: Collection,
    limit: int = 2
) -> List[Dict]:
    results = []

    search_params = {
        "metric_type": "IP",
        "params": {"nprobe": 10}
    }

    try:
        milvus_results = collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["name", "chunks", "action_link", "eGov_link", "mGov_link"]
        )

        for hits in milvus_results:
            for hit in hits:
                data = hit.entity
                link = data.get("action_link") or data.get("mGov_link") or data.get("eGov_link")

                results.append({
                    "name": data.get("name", ""),
                    "description": data.get("chunks", ""),
                    "link": link,
                    "score": hit.score,
                    "source": "egov_general_2_ru"
                })

        # Explicitly sort by score descending to guarantee consistency
        results.sort(key=lambda x: x["score"], reverse=True)

        # Log results (as you did previously)
        logger.info(f"Milvus search results: {results}")

    except Exception as e:
        logger.error(f"Milvus search error: {str(e)}")

    return results[:limit]

async def search_postgres(query: str, embedding: List[float], conn, limit: int = 2) -> List[Dict]:
    try:
        # Ensure embedding is a list
        try:
            embedding_list = embedding.tolist()  # works for numpy arrays
        except AttributeError:
            embedding_list = list(embedding)
        
        # Convert the embedding list into a string in the format pgvector expects
        embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'
        
        # Query the new egov_updated_2_ru table
        results = await conn.fetch("""
            SELECT 
                name, 
                chunks AS description, 
                "eGov link" AS egov_link, 
                "mGov link" AS mgov_link,
                action_link
            FROM egov_updated_2_ru
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """, embedding_str, limit)
        
        services = []
        for result in results:
            link = None
            if result['action_link']:
                link = result['action_link']
            elif result['mGov_link']:
                link = result['mGov_link']
            elif result['eGov_link']:
                link = result['eGov_link']

            services.append({
                "name": result['name'],
                "description": result['description'],
                "link": link,
                "source": "egov_updated_2_ru"
            })
        return services
    except Exception as e:
        logger.error(f"PostgreSQL extended search error: {str(e)}")
        return []
    
async def get_relevant_services(query: str, model, conn, collection: Collection) -> List[str]:
    embedding = await get_embedding_async(query, model)
    # Try Milvus search first
    if collection:
        milvus_services = await search_milvus(query, embedding, collection)
        if milvus_services:
            top_score = milvus_services[0]['score']
            logger.info(f"Found {len(milvus_services)} services in Milvus with top score: {top_score}")
            if top_score >= 0.18:
                return [
                    f"Name of the service corresponding to the chunk: {service['name']}\n"
                    f"Data about the service: {service['description']}\n"
                    f"eGov Mobile link of the service: {service['link']}"
                    for service in milvus_services
                ]
            else:
                logger.info(f"Top Milvus score < 0.18 ({top_score}), falling back to Postgres")
    # Fall back to PostgreSQL search    
    postgres_services = await search_postgres(query, embedding, conn)
    logger.info(f"Found {len(postgres_services)} services in PostgreSQL")
    return [
        f"Name of the service corresponding to the chunk: {service['name']}\n"
        f"Data about the service: {service['description']}\n"
        f"eGov Mobile link of the service: {service['link'] }"
        for service in postgres_services
    ]
# -----------------------
# Chat History Endpoints (Ephemeral Threads)
# -----------------------

# When a user starts a session, they provide their user_id.
# The thread ID is generated uniquely and the key in Redis is set to expire after CHAT_HISTORY_TTL seconds.
@app.post("/threads", response_model=dict)
async def create_thread(user_id: str = Body(..., example="user123"), redis_client=Depends(get_redis)):
    if not redis_client:
        raise HTTPException(status_code=503, detail=REDIS_UNAVLB_MSSG)
    try:
        thread_id = str(uuid.uuid4())
        key = f"chat_history:{user_id}:{thread_id}"
        # Create empty chat history and set TTL
        await redis_client.delete(key)
        await redis_client.rpush(key, json.dumps({"role": "system", "content": "Session started"}))
        await redis_client.expire(key, CHAT_HISTORY_TTL)
        
        return {"id": thread_id}
    except Exception as e:
        logger.error(f"Error creating thread: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create thread")

@app.post("/threads/{user_id}/{thread_id}/messages", response_model=dict)
async def add_message_to_thread(
    user_id: str,
    thread_id: str,
    message: dict = Body(..., example={"role": "user", "content": "Привет!"}),
    redis_client=Depends(get_redis)
):
    if not redis_client:
        raise HTTPException(status_code=503, detail=REDIS_UNAVLB_MSSG)
    try:
        key = f"chat_history:{user_id}:{thread_id}"
        # Check if thread exists
        exists = await redis_client.exists(key)
        if not exists:
            raise HTTPException(status_code=404, detail="Thread not found")
            
        await redis_client.rpush(key, json.dumps(message))
        # Refresh TTL on every message
        await redis_client.expire(key, CHAT_HISTORY_TTL)
        return {"status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding message to thread: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to add message")

@app.get("/threads/{user_id}/{thread_id}/messages", response_model=list)
async def get_thread_messages(user_id: str, thread_id: str, redis_client=Depends(get_redis)):
    if not redis_client:
        raise HTTPException(status_code=503, detail=REDIS_UNAVLB_MSSG)
    try:
        key = f"chat_history:{user_id}:{thread_id}"
        # Check if thread exists
        exists = await redis_client.exists(key)
        if not exists:
            raise HTTPException(status_code=404, detail="Thread not found")
            
        messages_raw = await redis_client.lrange(key, 0, -1)
        return [json.loads(m) for m in messages_raw]
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding message JSON: {str(e)}")
        raise HTTPException(status_code=500, detail="Corrupted message data")
    except Exception as e:
        logger.error(f"Error retrieving messages: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve messages")

@app.post("/threads/{user_id}/{thread_id}/runs")
async def run_thread(
    user_id: str,
    thread_id: str,
    request_data: EgovAiRequest,  # now accepts user_question and cache_response flag
    background_tasks: BackgroundTasks,
    redis_client=Depends(get_redis),
    conn=Depends(get_db_conn),
    model=Depends(get_model),
    collection=Depends(get_collection),
    httpx_client=Depends(get_httpx_client)
):
    if not redis_client:
        raise HTTPException(status_code=503, detail=REDIS_UNAVLB_MSSG)
    
    thread_key = f"chat_history:{user_id}:{thread_id}"
    # Append the user's new query to the thread history
    user_message = {"role": "user", "content": request_data.user_question}
    await redis_client.rpush(thread_key, json.dumps(user_message))
    await redis_client.expire(thread_key, CHAT_HISTORY_TTL)

    # If caching is enabled, check for an existing cached response
    if request_data.cache_response:
        cache_key = generate_cache_key(request_data.user_question)
        cached_response = await redis_client.get(cache_key)
        if cached_response:
            # Append the cached assistant response to the thread history
            await redis_client.rpush(thread_key, json.dumps({"role": "assistant", "content": cached_response}))
            await redis_client.expire(thread_key, CHAT_HISTORY_TTL)
            return {"id": thread_id, "assistant_message": cached_response}

    # Build the conversation with system prompt and current chat history
    messages_raw = await redis_client.lrange(thread_key, 0, -1)
    history = [json.loads(m) for m in messages_raw] if messages_raw else []

 # Get search results (context) based on the user's question
    search_results = await get_relevant_services(request_data.user_question, model, conn, collection)
    if search_results:
        search_results_text = "\n\n".join(search_results)
    else:
        search_results_text = "No relevant services found."

    # Build the conversation messages:
    # - The primary system prompt remains unchanged.
    # - A second system message carries the dynamic search result context.
    # - Then the conversation history follows.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Relevant services based on the query:\n\n{search_results_text}"}
    ] + history

    payload = {
        "model": "gpt-4.1-mini",
        "messages": messages,
    }
    payload_json = json.dumps(payload)

    try:
        response = await httpx_client.post(
            url="https://api.openai.com/v1/chat/completions",
            content=payload_json
        )
        response.raise_for_status()
        response_data = response.json()
        assistant_message = response_data['choices'][0]['message']['content']

        # Cache the response if caching is enabled
        if request_data.cache_response:
            cache_key = generate_cache_key(request_data.user_question)
            await redis_client.set(cache_key, assistant_message, ex=CHAT_HISTORY_TTL)

        # Append assistant's reply to chat history and refresh TTL
        await redis_client.rpush(thread_key, json.dumps({"role": "assistant", "content": assistant_message}))
        await redis_client.expire(thread_key, CHAT_HISTORY_TTL)
        return {"id": thread_id, "assistant_message": assistant_message}
    
    except httpx.HTTPStatusError as e:
        error_detail = f"OpenAI API error: {str(e)}"
        try:
            error_response = e.response.json()
            if "error" in error_response:
                error_detail = f"OpenAI API error: {error_response['error'].get('message', str(error_response['error']))}"
        except Exception:
            pass
        logger.error(error_detail)
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except Exception as e:
        logger.exception(f"Error processing run: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing run: {str(e)}")

# -----------------------
# Other Endpoints (health, metrics, admin, etc.)
# -----------------------

@app.get("/health")
async def health_check(
    redis_client=Depends(get_redis),
    collection=Depends(get_collection),
):
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "up",
            "redis": "unavailable",
            "zilliz": "unavailable",
            "database": "unavailable"
        }
    }
    if redis_client:
        try:
            await redis_client.ping()
            health_status["services"]["redis"] = "up"
        except Exception as e:
            health_status["services"]["redis"] = f"down: {str(e)}"
            health_status["status"] = "degraded"

    # Check if any collection is available and working
    if collection and any(collection.values()):
        try:
            # Try to access any valid collection
            for is_loaded in collection.items():
                if is_loaded:
                    health_status["services"]["zilliz"] = "up"
                    break
            else:
                health_status["services"]["zilliz"] = "down: no active collection"
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["services"]["zilliz"] = f"down: {str(e)}"
            health_status["status"] = "degraded"
    else:
        health_status["services"]["zilliz"] = "down: no collection"
        health_status["status"] = "degraded"
    try:
        async with app.state.db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
            health_status["services"]["database"] = "up"
    except Exception as e:
        health_status["services"]["database"] = f"down: {str(e)}"
        health_status["status"] = "degraded"
    return health_status

async def scan_keys(redis_client, pattern: str, count: int = 100):
    """Asynchronously yields keys matching the given pattern."""
    cursor = "0"
    while True:
        cursor, batch = await redis_client.scan(cursor=cursor, match=pattern, count=count)
        for key in batch:
            yield key
        if cursor == "0":
            break

# List active threads for a user
@app.get("/threads/{user_id}", response_model=list)
async def list_user_threads(user_id: str, redis_client=Depends(get_redis)):
    if not redis_client:
        raise HTTPException(status_code=503, detail=REDIS_UNAVLB_MSSG)
    try:
        pattern = f"chat_history:{user_id}:*"
        keys = []
        # For example, limit to 500 keys to avoid scanning too many entries
        max_keys = 500
        async for key in scan_keys(redis_client, pattern):
            keys.append(key)
            if len(keys) >= max_keys:
                break

        # Process keys to extract thread information (e.g., TTL, message count)
        threads = []
        for key in keys:
            thread_id = key.split(":")[-1]
            ttl = await redis_client.ttl(key)
            message_count = await redis_client.llen(key)
            if ttl > 0:
                last_message = None
                if message_count > 0:
                    last_msg_raw = await redis_client.lindex(key, -1)
                    try:
                        last_msg = json.loads(last_msg_raw)
                        last_message = last_msg.get("content", "")[:50]
                    except Exception:
                        pass

                threads.append({
                    "thread_id": thread_id,
                    "ttl_seconds": ttl,
                    "expires_in": f"{ttl//60} minutes" if ttl > 60 else f"{ttl} seconds",
                    "message_count": message_count,
                    "last_message": last_message
                })
        return threads
    except Exception as e:
        logger.error(f"Error listing threads: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list threads")

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5637))
    workers = int(os.getenv("WORKERS", min(os.cpu_count() or 1, 4)))
    logger.info(f"Starting server on {host}:{port} with {workers} workers")
    uvicorn.run(
        "mgov-apiv4-rest:app",
        host=host,
        port=port,
        workers=workers,
        http="httptools",
        log_level="info",
        reload=False,
        access_log=True,
        proxy_headers=True,
        timeout_keep_alive=65,
    )
