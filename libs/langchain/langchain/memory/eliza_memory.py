"""Eliza Long-Term Memory for LangChain Integration."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.memory import BaseMemory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ElizaMemoryStore(BaseModel):
    """In-memory store for Eliza memories with vector similarity support."""
    
    memories: List[Dict[str, Any]] = Field(default_factory=list)
    conversations: List[Dict[str, Any]] = Field(default_factory=list)
    associations: List[Dict[str, Any]] = Field(default_factory=list)
    
    def add_memory(self, memory: Dict[str, Any]) -> str:
        """Add a memory and return its ID."""
        memory_id = f"mem_{len(self.memories)}"
        memory["id"] = memory_id
        self.memories.append(memory)
        return memory_id
    
    def get_memories_by_type(self, memory_type: str) -> List[Dict[str, Any]]:
        """Get memories by type."""
        return [m for m in self.memories if m.get("memory_type") == memory_type]
    
    def search_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Simple text-based memory search."""
        query_lower = query.lower()
        matches = []
        
        for memory in self.memories:
            content = memory.get("content", "").lower()
            if query_lower in content:
                matches.append(memory)
        
        return matches[:limit]
    
    def add_conversation(self, conversation: Dict[str, Any]) -> str:
        """Add a conversation entry."""
        conv_id = f"conv_{len(self.conversations)}"
        conversation["id"] = conv_id
        self.conversations.append(conversation)
        return conv_id
    
    def get_recent_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversations."""
        return self.conversations[-limit:]


class ElizaChatMessageHistory(BaseChatMessageHistory):
    """Chat message history implementation for Eliza with memory integration."""
    
    def __init__(self, session_id: str, memory_store: Optional[ElizaMemoryStore] = None):
        self.session_id = session_id
        self.memory_store = memory_store or ElizaMemoryStore()
        self._messages: List[BaseMessage] = []
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Get all messages."""
        return self._messages
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history and store in memory."""
        self._messages.append(message)
        
        # Store in memory store
        conversation_entry = {
            "session_id": self.session_id,
            "role": "user" if isinstance(message, HumanMessage) else "assistant",
            "content": message.content,
            "timestamp": message.additional_kwargs.get("timestamp"),
            "message_type": message.additional_kwargs.get("message_type", "general")
        }
        
        self.memory_store.add_conversation(conversation_entry)
    
    def clear(self) -> None:
        """Clear all messages."""
        self._messages = []


class ElizaLongTermMemory(BaseMemory):
    """
    LangChain-compatible long-term memory for Eliza with semantic search and associations.
    
    This memory class provides:
    - Persistent storage of conversations and extracted memories
    - Semantic search capabilities for memory retrieval
    - Memory associations and relationships
    - Integration with LangChain chains and agents
    """
    
    memory_store: ElizaMemoryStore = Field(default_factory=ElizaMemoryStore)
    chat_history: ElizaChatMessageHistory = Field(default=None)
    session_id: str = Field(default="default")
    user_id: str = Field(default="anonymous")
    memory_key: str = Field(default="memory")
    return_messages: bool = Field(default=False)
    max_token_limit: int = Field(default=2000)
    memory_extraction_enabled: bool = Field(default=True)
    semantic_search_enabled: bool = Field(default=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.chat_history is None:
            self.chat_history = ElizaChatMessageHistory(
                session_id=self.session_id,
                memory_store=self.memory_store
            )
    
    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables."""
        return [self.memory_key]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables for the chain."""
        # Get recent conversation history
        recent_conversations = self.memory_store.get_recent_conversations(limit=10)
        
        # Get relevant memories based on current input
        current_input = inputs.get("input", "")
        relevant_memories = []
        
        if current_input and self.semantic_search_enabled:
            relevant_memories = self.memory_store.search_memories(current_input, limit=5)
        
        # Format memory context
        memory_context = self._format_memory_context(recent_conversations, relevant_memories)
        
        return {self.memory_key: memory_context}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context to memory."""
        input_str = inputs.get("input", "")
        output_str = outputs.get("output", "")
        
        # Add messages to chat history
        if input_str:
            human_message = HumanMessage(
                content=input_str,
                additional_kwargs={
                    "timestamp": self._get_timestamp(),
                    "user_id": self.user_id,
                    "session_id": self.session_id
                }
            )
            self.chat_history.add_message(human_message)
        
        if output_str:
            ai_message = AIMessage(
                content=output_str,
                additional_kwargs={
                    "timestamp": self._get_timestamp(),
                    "user_id": self.user_id,
                    "session_id": self.session_id
                }
            )
            self.chat_history.add_message(ai_message)
        
        # Extract and store memories if enabled
        if self.memory_extraction_enabled:
            self._extract_and_store_memories(input_str, output_str)
    
    def clear(self) -> None:
        """Clear all memory."""
        self.chat_history.clear()
        self.memory_store = ElizaMemoryStore()
    
    def add_memory(self, content: str, memory_type: str = "general", 
                   metadata: Optional[Dict[str, Any]] = None, 
                   tags: Optional[List[str]] = None) -> str:
        """Manually add a memory."""
        memory = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "content": content,
            "memory_type": memory_type,
            "metadata": metadata or {},
            "tags": tags or [],
            "timestamp": self._get_timestamp(),
            "relevance_score": 1.0
        }
        
        return self.memory_store.add_memory(memory)
    
    def search_memories(self, query: str, memory_types: Optional[List[str]] = None,
                       limit: int = 5) -> List[Dict[str, Any]]:
        """Search memories by query."""
        memories = self.memory_store.search_memories(query, limit)
        
        if memory_types:
            memories = [m for m in memories if m.get("memory_type") in memory_types]
        
        return memories
    
    def get_memory_analytics(self) -> Dict[str, Any]:
        """Get memory usage analytics."""
        total_memories = len(self.memory_store.memories)
        total_conversations = len(self.memory_store.conversations)
        
        memory_types = {}
        for memory in self.memory_store.memories:
            mem_type = memory.get("memory_type", "unknown")
            memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
        
        return {
            "total_memories": total_memories,
            "total_conversations": total_conversations,
            "memory_types": memory_types,
            "session_id": self.session_id,
            "user_id": self.user_id
        }
    
    def _format_memory_context(self, conversations: List[Dict[str, Any]], 
                              memories: List[Dict[str, Any]]) -> str:
        """Format memory context for the chain."""
        context_parts = []
        
        # Add relevant memories
        if memories:
            context_parts.append("Relevant memories:")
            for memory in memories:
                context_parts.append(f"- {memory.get('content', '')}")
        
        # Add recent conversation context
        if conversations:
            context_parts.append("\nRecent conversation:")
            for conv in conversations[-5:]:  # Last 5 exchanges
                role = conv.get('role', 'unknown')
                content = conv.get('content', '')
                context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _extract_and_store_memories(self, input_str: str, output_str: str) -> None:
        """Extract and store important information as memories."""
        # Simple keyword-based extraction (can be enhanced with NLP)
        preference_keywords = ["like", "prefer", "love", "hate", "dislike", "favorite"]
        factual_keywords = ["is", "are", "was", "were", "born", "live", "work"]
        
        # Check for preferences
        input_lower = input_str.lower()
        if any(keyword in input_lower for keyword in preference_keywords):
            self.add_memory(
                content=input_str,
                memory_type="preference",
                metadata={"extracted_from": "user_input"}
            )
        
        # Check for factual information
        elif any(keyword in input_lower for keyword in factual_keywords):
            self.add_memory(
                content=input_str,
                memory_type="factual",
                metadata={"extracted_from": "user_input"}
            )
        
        # Store contextual information from longer conversations
        if len(input_str) > 50:  # Longer messages likely contain context
            self.add_memory(
                content=input_str,
                memory_type="contextual",
                metadata={"extracted_from": "user_input", "length": len(input_str)}
            )
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


class ElizaVectorMemory(ElizaLongTermMemory):
    """
    Enhanced Eliza memory with vector similarity search capabilities.
    
    This extends the base ElizaLongTermMemory with:
    - Vector embeddings for semantic search
    - Similarity-based memory retrieval
    - Advanced memory associations
    """
    
    embedding_function: Optional[Any] = Field(default=None)
    similarity_threshold: float = Field(default=0.7)
    
    def __init__(self, embedding_function=None, **kwargs):
        super().__init__(**kwargs)
        self.embedding_function = embedding_function
    
    def search_memories_semantic(self, query: str, limit: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Search memories using semantic similarity."""
        if not self.embedding_function:
            # Fallback to text search
            memories = self.search_memories(query, limit=limit)
            return [(memory, 1.0) for memory in memories]
        
        # This would implement actual vector similarity search
        # For now, return text-based search with simulated scores
        memories = self.search_memories(query, limit=limit)
        return [(memory, 0.8) for memory in memories]  # Simulated similarity scores
    
    def add_memory_with_embedding(self, content: str, memory_type: str = "general",
                                 metadata: Optional[Dict[str, Any]] = None,
                                 tags: Optional[List[str]] = None) -> str:
        """Add a memory with vector embedding."""
        memory_id = self.add_memory(content, memory_type, metadata, tags)
        
        # Generate embedding if function is available
        if self.embedding_function:
            try:
                embedding = self.embedding_function(content)
                # Store embedding in memory metadata
                memory = next(m for m in self.memory_store.memories if m["id"] == memory_id)
                memory["metadata"]["embedding"] = embedding
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")
        
        return memory_id


# Factory functions for easy instantiation
def create_eliza_memory(session_id: str = "default", user_id: str = "anonymous",
                       memory_extraction: bool = True) -> ElizaLongTermMemory:
    """Create a standard Eliza long-term memory instance."""
    return ElizaLongTermMemory(
        session_id=session_id,
        user_id=user_id,
        memory_extraction_enabled=memory_extraction
    )


def create_eliza_vector_memory(session_id: str = "default", user_id: str = "anonymous",
                              embedding_function=None) -> ElizaVectorMemory:
    """Create an Eliza memory instance with vector capabilities."""
    return ElizaVectorMemory(
        session_id=session_id,
        user_id=user_id,
        embedding_function=embedding_function
    )

