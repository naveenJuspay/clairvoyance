"""
Daily Room Pool Manager

Manages a pool of pre-created Daily rooms with tokens to eliminate
the 1-second room creation delay on each connection.
"""

import asyncio
from typing import Dict, List, Optional
from asyncio import Queue

from pipecat.transports.daily.utils import DailyRESTHelper, DailyRoomParams

from app.core.logger import logger


class DailyRoom:
    """Represents a pre-created Daily room with tokens"""
    
    def __init__(self, room_url: str, user_token: str, bot_token: str):
        self.room_url = room_url
        self.user_token = user_token
        self.bot_token = bot_token
        self.is_used = False
        

class RoomPool:
    """Manages a pool of pre-created Daily rooms for quick allocation"""
    
    def __init__(self, daily_rest_helper: DailyRESTHelper, pool_size: int = 3, max_pool_size: int = 8):
        self.daily_rest_helper = daily_rest_helper
        self.pool_size = pool_size
        self.max_pool_size = max_pool_size
        self.available_rooms: Queue[DailyRoom] = Queue()
        self.active_rooms: Dict[str, DailyRoom] = {}  # session_id -> room
        self._create_lock = asyncio.Lock()
        self.is_creating_room = False
        
    async def initialize(self):
        """Create initial pool of rooms"""
        logger.info(f"Initializing Daily room pool with {self.pool_size} rooms")
        
        for i in range(self.pool_size):
            try:
                await self._create_and_add_room()
                logger.info(f"Created room {i+1}/{self.pool_size}")
            except Exception as e:
                logger.error(f"Failed to create room {i+1}: {e}")
                
        logger.info(f"Daily room pool initialized with {self.available_rooms.qsize()} rooms")
        
    async def _create_and_add_room(self):
        """Create a new room with tokens and add to pool"""
        try:
            # Create room
            room = await self.daily_rest_helper.create_room(DailyRoomParams())
            if not room.url:
                raise RuntimeError("Failed to create room - no URL returned")

            # Get user token
            user_token = await self.daily_rest_helper.get_token(room.url)
            if not user_token:
                raise RuntimeError("Failed to get user token")

            # Get bot token
            bot_token = await self.daily_rest_helper.get_token(room.url)
            if not bot_token:
                raise RuntimeError("Failed to get bot token")

            # Create room object and add to pool
            daily_room = DailyRoom(room.url, user_token, bot_token)
            await self.available_rooms.put(daily_room)
            
            logger.info(f"Created and added room to pool: {room.url}")
            
        except Exception as e:
            logger.error(f"Error creating room for pool: {e}")
            raise
            
    async def get_room(self, session_id: str) -> DailyRoom:
        """Get an available room from the pool"""
        logger.info(f"Getting room for session {session_id}")
        
        try:
            # Get room from pool with short timeout
            room = await asyncio.wait_for(
                self.available_rooms.get(),
                timeout=0.1
            )
            
            # Mark as active
            room.is_used = True
            self.active_rooms[session_id] = room
            
            # Start background room creation if pool is getting low
            if (self.available_rooms.qsize() <= 1 and
                not self.is_creating_room):
                logger.info("Room pool getting low, creating background room")
                asyncio.create_task(self._create_background_room())
                
            logger.info(f"Assigned room {room.room_url} to session {session_id}")
            return room
            
        except asyncio.TimeoutError:
            # Pool exhausted - create room directly
            logger.warning(f"Room pool exhausted for session {session_id}, creating room directly")
            return await self._create_room_direct(session_id)
            
    async def _create_background_room(self):
        """Create a new room in the background"""
        async with self._create_lock:
            if self.is_creating_room:
                return
            self.is_creating_room = True
            
        try:
            await self._create_and_add_room()
            logger.info("Background room created successfully")
        except Exception as e:
            logger.error(f"Failed to create background room: {e}")
        finally:
            async with self._create_lock:
                self.is_creating_room = False
                
    async def _create_room_direct(self, session_id: str) -> DailyRoom:
        """Create a room directly for immediate use (fallback)"""
        logger.info(f"Creating direct room for session {session_id}")
        
        try:
            # Create room
            room = await self.daily_rest_helper.create_room(DailyRoomParams())
            if not room.url:
                raise RuntimeError("Failed to create room - no URL returned")

            # Get user token
            user_token = await self.daily_rest_helper.get_token(room.url)
            if not user_token:
                raise RuntimeError("Failed to get user token")

            # Get bot token  
            bot_token = await self.daily_rest_helper.get_token(room.url)
            if not bot_token:
                raise RuntimeError("Failed to get bot token")

            # Create room object and mark as active immediately
            daily_room = DailyRoom(room.url, user_token, bot_token)
            daily_room.is_used = True
            self.active_rooms[session_id] = daily_room
            
            logger.info(f"Direct room created for session {session_id}: {room.url}")
            return daily_room
            
        except Exception as e:
            logger.error(f"Failed to create direct room for session {session_id}: {e}")
            raise
            
    async def return_room(self, session_id: str):
        """Delete room after session ends and replenish pool"""
        if session_id not in self.active_rooms:
            logger.warning(f"Session {session_id} not found in active rooms")
            return
            
        room = self.active_rooms.pop(session_id)
        
        # Delete the room (Daily rooms are single-use)
        try:
            await self.delete_room(room.room_url)
            logger.info(f"Deleted room for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to delete room for session {session_id}: {e}")
        
        # Create replacement room if needed
        if self.available_rooms.qsize() < self.pool_size and not self.is_creating_room:
            asyncio.create_task(self._create_background_room())
            
    async def delete_room(self, room_url: str):
        """Delete a Daily room"""
        try:
            await self.daily_rest_helper.delete_room_by_url(room_url)
        except Exception as e:
            logger.error(f"Error deleting room {room_url}: {e}")
            raise
            
    async def get_pool_stats(self) -> Dict:
        """Get current room pool statistics"""
        return {
            "available_rooms": self.available_rooms.qsize(),
            "active_rooms": len(self.active_rooms),
            "is_creating_room": self.is_creating_room,
            "pool_size": self.pool_size,
            "max_pool_size": self.max_pool_size
        }
        
    async def cleanup(self):
        """Clean up all rooms in the pool"""
        logger.info("Cleaning up Daily room pool")
        
        # Delete all available rooms
        while not self.available_rooms.empty():
            try:
                room = self.available_rooms.get_nowait()
                await self.delete_room(room.room_url)
            except asyncio.QueueEmpty:
                break
                
        # Delete all active rooms
        for session_id, room in list(self.active_rooms.items()):
            await self.delete_room(room.room_url)
            
        self.active_rooms.clear()
        logger.info("Daily room pool cleanup complete")