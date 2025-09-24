"""
Voice Agent Process Pool Manager

Manages a pool of pre-warmed voice agent processes to eliminate
the 5-6 second initialization delay on each connection.
"""

import asyncio
import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from asyncio import Queue

from app.core.logger import logger


class VoiceAgentProcess:
    """Represents a single voice agent process in the pool"""
    
    def __init__(self, process: asyncio.subprocess.Process, process_id: str):
        self.process = process
        self.process_id = process_id
        self.is_busy = False
        self.session_id: Optional[str] = None
        self.created_at = datetime.now()
        self.last_used_at: Optional[datetime] = None
        
    def mark_busy(self, session_id: str):
        """Mark process as busy with a session"""
        self.is_busy = True
        self.session_id = session_id
        self.last_used_at = datetime.now()
        
    def mark_available(self):
        """Mark process as available for new sessions"""
        self.is_busy = False
        self.session_id = None
        
    def is_healthy(self) -> bool:
        """Check if the process is still running and healthy"""
        return self.process.returncode is None


class VoiceAgentPool:
    """Manages a pool of pre-warmed voice agent processes"""
    
    def __init__(self, pool_size: int = 2, max_pool_size: int = 5):
        self.pool_size = pool_size
        self.max_pool_size = max_pool_size
        self.available_processes: Queue[VoiceAgentProcess] = Queue()
        self.active_processes: Dict[str, VoiceAgentProcess] = {}
        self.all_processes: Dict[str, VoiceAgentProcess] = {}
        self._create_lock = asyncio.Lock()
        self.is_creating_process = False
        
    async def initialize(self):
        """Create initial pool of processes"""
        logger.info(f"Initializing voice agent pool with {self.pool_size} processes")
        
        for i in range(self.pool_size):
            try:
                await self._create_and_add_process()
                logger.info(f"Created process {i+1}/{self.pool_size}")
            except Exception as e:
                logger.error(f"Failed to create process {i+1}: {e}")
                
        logger.info(f"Voice agent pool initialized with {len(self.all_processes)} processes")
        
    async def _create_and_add_process(self):
        """Create a new pre-warmed process and add to pool"""
        process_id = str(uuid.uuid4())
        
        try:
            # Create the subprocess in pool mode with unbuffered output
            cmd = f"python3 -u -m app.agents.voice.automatic --pool-mode --process-id {process_id}"
            
            proc = await asyncio.create_subprocess_shell(
                cmd,
                cwd=Path(__file__).parent.parent.parent,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
                env=os.environ
            )
            
            voice_process = VoiceAgentProcess(proc, process_id)
            self.all_processes[process_id] = voice_process
            
            # Wait for process to signal it's ready (with timeout)
            await self._wait_for_process_ready(voice_process)
            
            # Always monitor for session end, but only forward logs in development
            asyncio.create_task(self._monitor_process_output(voice_process))
            
            # Add to available pool
            await self.available_processes.put(voice_process)
            
            logger.info(f"Created and added process {process_id} to pool")
            
        except Exception as e:
            logger.error(f"Failed to create process {process_id}: {e}")
            # Clean up if process was created but failed
            if process_id in self.all_processes:
                await self._cleanup_process(process_id)
            raise
            
    async def _wait_for_process_ready(self, voice_process: VoiceAgentProcess, timeout: int = 30):
        """Wait for process to signal it's ready"""
        loop = asyncio.get_running_loop()
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            # Check if process is still running
            if not voice_process.is_healthy():
                raise RuntimeError(f"Process {voice_process.process_id} died during startup")
            
            try:
                # Try to read a line with a short timeout
                ready_line = await asyncio.wait_for(
                    voice_process.process.stdout.readline(),
                    timeout=0.5
                )
                
                if ready_line:
                    ready_line = ready_line.decode('utf-8') if isinstance(ready_line, bytes) else ready_line
                
                if ready_line:
                    logger.debug(f"Process {voice_process.process_id} output: {ready_line.strip()}")
                    if "READY" in ready_line:
                        logger.info(f"Process {voice_process.process_id} signaled READY")
                        return
                        
            except asyncio.TimeoutError:
                # Continue waiting
                await asyncio.sleep(0.1)
                continue
        
        # Timeout reached - stderr is merged into stdout, so no separate error capture needed
            
        raise TimeoutError(f"Process {voice_process.process_id} timed out waiting for READY signal")
    
    async def _monitor_process_output(self, voice_process: VoiceAgentProcess):
        """Monitor subprocess output for session end signal and forward logs in dev"""
        is_dev = os.getenv("ENVIRONMENT", "production").lower() in ["dev", "development"]
        
        try:
            while voice_process.is_healthy():
                try:
                    line = await asyncio.wait_for(
                        voice_process.process.stdout.readline(),
                        timeout=2.0
                    )
                    if line:
                        line = line.decode('utf-8').strip() if isinstance(line, bytes) else line.strip()
                        
                        if line:
                            # Always check for session end signal (it might be concatenated with other text)
                            if "SESSION_ENDED" in line:
                                logger.info(f"Process {voice_process.process_id[:8]} session ended, returning to pool")
                                await self._return_process_to_pool(voice_process)
                            elif is_dev:
                                # Forward logs only in development
                                logger.info(f"[Process {voice_process.process_id[:8]}] {line}")
                            
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.debug(f"Process monitoring error for {voice_process.process_id}: {e}")
                    break
        except Exception as e:
            logger.debug(f"Process monitoring stopped for {voice_process.process_id}: {e}")
    
    async def _return_process_to_pool(self, voice_process: VoiceAgentProcess):
        """Return a process to the pool after session ends"""
        try:
            # Find the session ID for this process
            session_id = None
            for sid, proc in self.active_processes.items():
                if proc.process_id == voice_process.process_id:
                    session_id = sid
                    break
            
            if session_id:
                await self.return_process(session_id)
                logger.info(f"Process {voice_process.process_id[:8]} returned to pool automatically")
                
                # Trigger immediate room cleanup
                await self._trigger_room_cleanup(session_id)
                
            else:
                logger.warning(f"Could not find session for process {voice_process.process_id[:8]}")
        except Exception as e:
            logger.error(f"Error returning process {voice_process.process_id[:8]} to pool: {e}")
            
    async def _trigger_room_cleanup(self, session_id: str):
        """Trigger room cleanup for a session"""
        try:
            # Import here to avoid circular imports
            from app.main import room_pool, cleanup_session_tracking, bot_procs
            
            if room_pool:
                await room_pool.return_room(session_id)
                logger.info(f"Triggered room cleanup for session {session_id}")
            
            # Also trigger session tracking cleanup
            for pid, proc_info in list(bot_procs.items()):
                if len(proc_info) >= 3 and proc_info[2] == session_id:
                    await cleanup_session_tracking(pid)
                    break
                    
        except Exception as e:
            logger.error(f"Error triggering room cleanup for session {session_id}: {e}")
            
    async def get_process(self, session_id: str) -> VoiceAgentProcess:
        """Get an available process for a session"""
        logger.info(f"Getting process for session {session_id}")
        
        try:
            # Try to drain unhealthy processes until we find a healthy one
            for _ in range(max(1, self.available_processes.qsize())):
                process = await asyncio.wait_for(
                    self.available_processes.get(),
                    timeout=0.1
                )
                
                # Check if process is healthy
                if process.is_healthy():
                    # Mark as active
                    process.mark_busy(session_id)
                    self.active_processes[session_id] = process
                    
                    # Start background process creation if pool is getting low
                    if (self.available_processes.qsize() == 0 and
                        len(self.all_processes) < self.max_pool_size and
                        not self.is_creating_process):
                        logger.info("Pool getting low, creating background process")
                        asyncio.create_task(self._create_background_process())
                        
                    logger.info(f"Assigned process {process.process_id} to session {session_id}")
                    return process
                else:
                    # Unhealthy process - cleanup and keep trying
                    logger.warning(f"Process {process.process_id} is unhealthy, cleaning up")
                    await self._cleanup_process(process.process_id)
                    
        except asyncio.TimeoutError:
            pass
            
        # Pool exhausted or no healthy processes - create directly (fallback)
        logger.warning(f"Pool exhausted for session {session_id}, creating process directly")
        return await self._create_process_direct(session_id)
            
    async def _create_background_process(self):
        """Create a new process in the background"""
        async with self._create_lock:
            if self.is_creating_process:
                return
            self.is_creating_process = True
            
        try:
            await self._create_and_add_process()
            logger.info("Background process created successfully")
        except Exception as e:
            logger.error(f"Failed to create background process: {e}")
        finally:
            async with self._create_lock:
                self.is_creating_process = False
            
    async def _create_process_direct(self, session_id: str) -> VoiceAgentProcess:
        """Create a process directly for immediate use (fallback)"""
        logger.info(f"Creating direct process for session {session_id}")
        
        process_id = str(uuid.uuid4())
        
        try:
            # Create the subprocess in pool mode with unbuffered output
            cmd = f"python3 -u -m app.agents.voice.automatic --pool-mode --process-id {process_id}"
            
            proc = await asyncio.create_subprocess_shell(
                cmd,
                cwd=Path(__file__).parent.parent.parent,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
                env=os.environ
            )
            
            voice_process = VoiceAgentProcess(proc, process_id)
            self.all_processes[process_id] = voice_process
            
            # Wait for process to signal it's ready
            await self._wait_for_process_ready(voice_process)
            
            # Always monitor for session end, but only forward logs in development
            asyncio.create_task(self._monitor_process_output(voice_process))
            
            # Mark as busy and active immediately
            voice_process.mark_busy(session_id)
            self.active_processes[session_id] = voice_process
            
            logger.info(f"Direct process {process_id} created for session {session_id}")
            return voice_process
            
        except Exception as e:
            logger.error(f"Failed to create direct process {process_id}: {e}")
            # Clean up if process was created but failed
            if process_id in self.all_processes:
                await self._cleanup_process(process_id)
            raise
        
    async def return_process(self, session_id: str):
        """Return process to pool after session ends"""
        if session_id not in self.active_processes:
            logger.warning(f"Session {session_id} not found in active processes")
            return
            
        process = self.active_processes.pop(session_id)
        
        # Check if process is still healthy
        if process.is_healthy():
            process.mark_available()
            await self.available_processes.put(process)
            logger.info(f"Returned process {process.process_id} to pool")
            
            # Note: Session tracking cleanup will be handled by the main app
            # when it detects the SESSION_ENDED signal
            
        else:
            logger.warning(f"Process {process.process_id} is unhealthy, removing from pool")
            await self._cleanup_process(process.process_id)
            
            # Create replacement process in background if we're below pool size
            # or if we have no available processes
            if (len(self.all_processes) < self.pool_size or
                (self.available_processes.qsize() == 0 and
                 len(self.all_processes) < self.max_pool_size)):
                asyncio.create_task(self._create_background_process())
                
    async def _cleanup_process(self, process_id: str):
        """Clean up a process and remove from all tracking"""
        if process_id not in self.all_processes:
            return
            
        process = self.all_processes.pop(process_id)
        
        try:
            if process.is_healthy():
                process.process.terminate()
                # Wait for process to terminate
                try:
                    await asyncio.wait_for(
                        process.process.wait(),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Process {process_id} did not terminate, killing")
                    process.process.kill()
                    await process.process.wait()
                    
        except Exception as e:
            logger.error(f"Error cleaning up process {process_id}: {e}")
            
        logger.info(f"Cleaned up process {process_id}")
        
    # Removed _wait_for_process_termination - using process.wait() directly
            
    async def get_pool_stats(self) -> Dict:
        """Get current pool statistics"""
        return {
            "total_processes": len(self.all_processes),
            "available_processes": self.available_processes.qsize(),
            "active_processes": len(self.active_processes),
            "is_creating_process": self.is_creating_process,
            "pool_size": self.pool_size,
            "max_pool_size": self.max_pool_size
        }
        
    async def cleanup(self):
        """Clean up all processes in the pool"""
        logger.info("Cleaning up voice agent pool")
        
        # Clean up all processes
        for process_id in list(self.all_processes.keys()):
            await self._cleanup_process(process_id)
            
        # Clear queues
        await self._purge_available_queue()
        self.active_processes.clear()
        logger.info("Voice agent pool cleanup complete")
        
    async def _purge_available_queue(self):
        """Remove stale processes from available queue"""
        tmp = []
        while not self.available_processes.empty():
            try:
                p = self.available_processes.get_nowait()
                if p.process_id in self.all_processes and p.is_healthy() and not p.is_busy:
                    tmp.append(p)
            except asyncio.QueueEmpty:
                break
        for p in tmp:
            await self.available_processes.put(p)


# Global pool instance
voice_agent_pool: Optional[VoiceAgentPool] = None


def get_voice_agent_pool() -> VoiceAgentPool:
    """Get the global voice agent pool instance"""
    global voice_agent_pool
    if voice_agent_pool is None:
        voice_agent_pool = VoiceAgentPool()
    return voice_agent_pool


async def initialize_voice_agent_pool(pool_size: int = 2, max_pool_size: int = 5):
    """Initialize the global voice agent pool"""
    global voice_agent_pool
    voice_agent_pool = VoiceAgentPool(pool_size=pool_size, max_pool_size=max_pool_size)
    await voice_agent_pool.initialize()


async def cleanup_voice_agent_pool():
    """Cleanup the global voice agent pool"""
    global voice_agent_pool
    if voice_agent_pool:
        await voice_agent_pool.cleanup()
        voice_agent_pool = None