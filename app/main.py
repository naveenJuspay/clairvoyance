import json
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pipecat.transports.daily.utils import (
    DailyMeetingTokenParams,
    DailyMeetingTokenProperties,
    DailyRESTHelper,
    DailyRoomParams,
    DailyRoomProperties,
)

from app import __version__
from app.api.routers import breeze_buddy
from app.core.config import (
    DAILY_API_KEY,
    DAILY_API_URL,
    ENABLE_AUTOMATIC_DAILY_RECORDING,
    HOST,
    MAX_DAILY_SESSION_LIMIT,
    PORT,
)

# Import necessary components from the new structure
from app.core.logger import logger
from app.core.security.jwt import validate_breeze_user
from app.core.transport.http_client import create_aiohttp_session
from app.core.process_pool import (
    cleanup_voice_agent_pool,
    get_voice_agent_pool,
    initialize_voice_agent_pool,
)
from app.core.room_pool import RoomPool

# Database imports
from app.database import close_db_pool, get_db_connection, init_db_pool
from app.schemas import (
    AutomaticVoiceUserConnectRequest,
)

# Dictionary to track bot processes: {pid: (process, room_url, session_id, proc_type)}
bot_procs = {}

# Store Daily API helpers and room pool
daily_helpers = {}
room_pool: Optional[RoomPool] = None


async def cleanup_session_tracking(pid: int):
    """Remove process from tracking when returned to pool"""
    if pid in bot_procs:
        proc_info = bot_procs.pop(pid)
        if len(proc_info) >= 3:
            _, _, session_id, proc_type = proc_info[:4]
            logger.info(f"Cleaned up session tracking for PID {pid} (session: {session_id})")
        else:
            logger.info(f"Cleaned up legacy session tracking for PID {pid}")


async def monitor_session_cleanup():
    """Monitor for completed sessions and clean up tracking"""
    import asyncio
    
    while True:
        try:
            # Clean up tracking for terminated processes
            pids_to_remove = []
            for pid, proc_info in bot_procs.items():
                if len(proc_info) >= 4:
                    proc, room_url, session_id, proc_type = proc_info
                    
                    # Check if process is still running
                    if hasattr(proc, 'poll'):
                        if proc.poll() is not None:  # Process terminated
                            pids_to_remove.append(pid)
                    elif hasattr(proc, 'returncode'):
                        if proc.returncode is not None:  # Process terminated
                            pids_to_remove.append(pid)
            
            # Remove completed sessions from tracking
            for pid in pids_to_remove:
                await cleanup_session_tracking(pid)
                
        except Exception as e:
            logger.debug(f"Error in session cleanup monitor: {e}")
        
        # Check every 10 seconds (reduced frequency)
        await asyncio.sleep(10)


def cleanup():
    """Cleanup function to terminate all bot processes.

    Called during server shutdown.
    """
    logger.info(f"Attempting to terminate {len(bot_procs)} bot processes.")
    for pid, proc_info in list(bot_procs.items()):
        try:
            if len(proc_info) == 4:
                proc, room_url, session_id, proc_type = proc_info
            else:
                # Legacy format
                proc, room_url = proc_info[:2]
                session_id = "unknown"
                proc_type = "legacy"
                
            if hasattr(proc, 'poll'):
                # subprocess.Popen
                is_running = proc.poll() is None
            else:
                # asyncio.subprocess.Process
                is_running = proc.returncode is None
                
            if is_running:
                logger.info(f"Terminating process {pid} for room {room_url} (session: {session_id})...")
                proc.terminate()
                if hasattr(proc, 'wait'):
                    proc.wait()
                logger.info(f"Process {pid} terminated successfully.")
            else:
                logger.info(
                    f"Process {pid} for room {room_url} has already terminated."
                )
        except Exception as e:
            logger.error(f"Error terminating process {pid}: {e}", exc_info=True)
        finally:
            # Ensure the process is removed from the tracking dictionary
            bot_procs.pop(pid, None)
    logger.info("All bot processes have been handled.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager that handles startup and shutdown tasks."""
    global room_pool
    logger.info("Application startup...")

    # Initialize database and create tables if needed
    try:
        await init_db_pool()
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

    # Initialize aiohttp session with proxy support for Daily API
    aiohttp_session = create_aiohttp_session()
    daily_helpers["rest"] = DailyRESTHelper(
        daily_api_key=DAILY_API_KEY,
        daily_api_url=DAILY_API_URL,
        aiohttp_session=aiohttp_session,
    )
    logger.info("Daily REST helper initialized with proxy support.")

    # Initialize Daily room pool
    try:
        room_pool = RoomPool(daily_helpers["rest"], pool_size=3, max_pool_size=8)
        await room_pool.initialize()
        logger.info("Daily room pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize room pool: {e}")

    # Initialize voice agent process pool
    try:
        await initialize_voice_agent_pool(pool_size=2, max_pool_size=5)
        logger.info("Voice agent process pool initialized")
        
        # Start background task to monitor session cleanup
        import asyncio
        asyncio.create_task(monitor_session_cleanup())
        
    except Exception as e:
        logger.error(f"Failed to initialize voice agent pool: {e}")

    yield

    logger.info("Application shutdown event triggered...")
    # Cleanup room pool
    if room_pool:
        await room_pool.cleanup()
    # Cleanup voice agent pool
    await cleanup_voice_agent_pool()
    # Cleanup bot processes
    cleanup()
    # Close database pool
    await close_db_pool()
    # Close aiohttp session
    await aiohttp_session.close()
    logger.info("Aiohttp session closed.")


app = FastAPI(title="Breeze Automatic Server", version=__version__, lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(
    breeze_buddy.router, prefix="/agent/voice/breeze-buddy", tags=["Breeze Buddy"]
)


# Pipecat bot endpoint
@app.post("/agent/voice/automatic")
async def bot_connect(
    request: AutomaticVoiceUserConnectRequest,
    user_context=Depends(validate_breeze_user),
) -> Dict[str, Any]:
    logger.info(
        f"Received new user connect request payload: {request.model_dump_json(exclude_none=True)}"
    )

    if user_context:
        logger.info(
            f"Authenticated user: {user_context['email']} (merchant: {user_context['merchantId']})"
        )

    # 1. Validate request
    raw_mode = request.mode
    euler_tok = request.eulerToken
    breeze_tok = request.breezeToken
    shop_url = request.shopUrl
    shop_id = request.shopId
    shop_type = request.shopType
    user_email = request.email
    user_name = request.userName
    tts_provider = request.ttsService.ttsProvider.value if request.ttsService else None
    voice_name = request.ttsService.voiceName.value if request.ttsService else None
    merchant_id = request.merchantId
    platform_integrations = request.platformIntegrations
    reseller_id = request.resellerId

    # 2. Get pre-created room from pool
    try:
        # Generate unique session ID first
        session_id = str(uuid.uuid4())
        
        # Get room from pool (this is now ~0.1s instead of ~1s)
        daily_room = await room_pool.get_room(session_id)
        room_url = daily_room.room_url
        token = daily_room.user_token
        bot_token = daily_room.bot_token
        
        logger.info(f"Got pre-created room from pool for session {session_id}: {room_url}")
        
    except Exception as e:
        logger.warning(f"Failed to get room from pool: {e}, falling back to direct creation")
        
        # Fallback: Create room directly (original method)
        daily_room_properties = DailyRoomProperties(
            exp=time.time() + MAX_DAILY_SESSION_LIMIT,
            eject_at_room_exp=True,
        )

        # Enable recording only if configured
        if ENABLE_AUTOMATIC_DAILY_RECORDING:
            daily_room_properties.enable_recording = "cloud"

        room = await daily_helpers["rest"].create_room(
            params=DailyRoomParams(properties=daily_room_properties)
        )

        token_params = DailyMeetingTokenParams(
            properties=DailyMeetingTokenProperties(
                eject_after_elapsed=MAX_DAILY_SESSION_LIMIT,
            )
        )

        token = await daily_helpers["rest"].get_token(
            room.url,
            expiry_time=MAX_DAILY_SESSION_LIMIT,
            eject_at_token_exp=True,
            owner=True,
            params=token_params,
        )
        
        room_url = room.url
        bot_token = token  # For fallback, use same token for bot
        session_id = str(uuid.uuid4())

    # 3. Generate client session ID for this subprocess
    client_sid = request.sessionId or str(
        uuid.uuid4()
    )  # Use client-provided sessionId or generate fallback
    logger.bind(session_id=session_id).info(
        f"Using session ID for new voice agent: {session_id}"
    )
    logger.bind(client_sid=client_sid).info(
        f"Using client session ID for new voice agent: {client_sid}"
    )

    # 4. Try to get process from pool first
    pool = get_voice_agent_pool()
    try:
        voice_process = await pool.get_process(session_id)
        
        # Configure the pre-warmed process for this session
        session_config = {
            "room_url": room_url,
            "token": bot_token,  # Use bot token for the voice agent
            "session_id": session_id,
            "client_sid": client_sid,
            "mode": raw_mode.upper() if raw_mode else None,
            "user_name": user_name,
            "user_email": user_email,
            "tts_provider": tts_provider,
            "voice_name": voice_name,
            "euler_token": euler_tok,
            "breeze_token": breeze_tok,
            "shop_url": shop_url,
            "shop_id": shop_id,
            "shop_type": shop_type,
            "merchant_id": merchant_id,
            "platform_integrations": platform_integrations,
            "reseller_id": reseller_id,
        }
        
        # Send config to process via stdin
        config_json = json.dumps(session_config) + "\n"
        voice_process.process.stdin.write(config_json.encode('utf-8'))
        await voice_process.process.stdin.drain()
        
        logger.bind(session_id=session_id).info(
            f"Assigned pre-warmed process {voice_process.process_id} to session {session_id}"
        )
        
        # Track the process for cleanup with session info
        bot_procs[voice_process.process.pid] = (voice_process.process, room_url, session_id, "pool")
        
        return {"room_url": room_url, "token": token, "session_id": session_id}
        
    except Exception as e:
        logger.warning(f"Failed to get process from pool: {e}, falling back to direct creation")
        
        # 5. Fallback: Launch subprocess without shell (original method)
        bot_file = "app.agents.voice.automatic"
        cmd = [
            "python3",
            "-m",
            bot_file,
            "-u",
            room_url,
            "-t",
            bot_token,
            "--mode",
            raw_mode.upper() if raw_mode else None,
            "--session-id",
            session_id,
            "--client-sid",
            client_sid,
        ]

        # Add user_name and tts_service regardless of mode
        if user_name:
            cmd += ["--user-name", user_name]
        if user_email:
            cmd += ["--user-email", user_email]
        if tts_provider:
            cmd += ["--tts-provider", tts_provider]
        if voice_name:
            cmd += ["--voice-name", voice_name]
        if euler_tok:
            cmd += ["--euler-token", euler_tok]
        if breeze_tok:
            cmd += ["--breeze-token", breeze_tok]
        if shop_url:
            cmd += ["--shop-url", shop_url]
        if shop_id:
            cmd += ["--shop-id", shop_id]
        if shop_type:
            cmd += ["--shop-type", shop_type]
        if merchant_id:
            cmd += ["--merchant-id", merchant_id]
        if platform_integrations:
            cmd += ["--platform-integrations"] + platform_integrations

        if reseller_id:
            cmd += ["--reseller-id", reseller_id]

        logger.bind(session_id=session_id).info(
            f"Launching subprocess with command: {' '.join(cmd)}"
        )
        proc = subprocess.Popen(
            cmd,
            cwd=Path(__file__).parent.parent,
            bufsize=1,
        )
        bot_procs[proc.pid] = (proc, room_url, session_id, "direct")
        logger.bind(session_id=session_id).info(f"Subprocess started with PID: {proc.pid}")

        return {"room_url": room_url, "token": token, "session_id": session_id}


# Serve client.html at the root
@app.get("/")
async def get_client_html():
    return FileResponse("static/home.html")


# Health check endpoint
@app.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    return JSONResponse({"status": "healthy"})


# Database health check endpoint
@app.get("/health/database")
async def database_health_check():
    """Check database connectivity and health."""
    logger.info("Database health check endpoint called")
    try:
        async for conn in get_db_connection():
            result = await conn.fetchval("SELECT 1")
            if result == 1:
                return JSONResponse(
                    {
                        "status": "healthy",
                        "database": "connected",
                        "message": "Database connection is healthy",
                    }
                )
            else:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "unhealthy",
                        "database": "error",
                        "message": "Database query returned unexpected result",
                    },
                )
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "status": "unhealthy",
                "database": "disconnected",
                "message": f"Database connection failed: {str(e)}",
            },
        )


# Version endpoint
@app.get("/version")
async def get_version():
    """Get application version."""
    return JSONResponse({"version": __version__})


# Pool status endpoint
@app.get("/pool/status")
async def get_pool_status():
    """Get voice agent and room pool status."""
    try:
        pool = get_voice_agent_pool()
        voice_stats = await pool.get_pool_stats()
        
        room_stats = {}
        if room_pool:
            room_stats = await room_pool.get_pool_stats()
        
        return JSONResponse({
            "status": "healthy",
            "voice_pool_stats": voice_stats,
            "room_pool_stats": room_stats
        })
    except Exception as e:
        logger.error(f"Error getting pool status: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


# Room pool status endpoint
@app.get("/pool/rooms/status")
async def get_room_pool_status():
    """Get Daily room pool status."""
    try:
        if not room_pool:
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable", "message": "Room pool not initialized"}
            )
            
        stats = await room_pool.get_pool_stats()
        return JSONResponse({"status": "healthy", "room_pool_stats": stats})
    except Exception as e:
        logger.error(f"Error getting room pool status: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


# Session cleanup endpoint
@app.post("/agent/voice/automatic/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    """Cleanup a specific voice agent session."""
    try:
        pool = get_voice_agent_pool()
        
        # Find and cleanup the session
        for pid, proc_info in list(bot_procs.items()):
            if len(proc_info) >= 3 and proc_info[2] == session_id:
                proc, room_url, sess_id, proc_type = proc_info
                
                logger.info(f"Cleaning up session {session_id} (PID: {pid}, type: {proc_type})")
                
                # Clean up room first
                if room_pool:
                    await room_pool.return_room(session_id)
                
                # If it's a pool process, return it to the pool
                if proc_type == "pool":
                    await pool.return_process(session_id)
                    logger.info(f"Returned process to pool for session {session_id}")
                else:
                    # Direct process - terminate it
                    try:
                        if hasattr(proc, 'poll'):
                            if proc.poll() is None:
                                proc.terminate()
                                proc.wait()
                        else:
                            if proc.returncode is None:
                                proc.terminate()
                                await proc.wait()
                        logger.info(f"Terminated direct process for session {session_id}")
                    except Exception as e:
                        logger.error(f"Error terminating process for session {session_id}: {e}")
                
                # Remove from tracking
                bot_procs.pop(pid, None)
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Session {session_id} cleaned up successfully",
                    "process_type": proc_type
                })
        
        return JSONResponse({
            "status": "not_found",
            "message": f"Session {session_id} not found"
        })
        
    except Exception as e:
        logger.error(f"Error cleaning up session {session_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )




# The main block is now only for direct execution, which is not the recommended way.
# Uvicorn running from run.py is the standard.
if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
