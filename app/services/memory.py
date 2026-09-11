from typing import List, Dict
from collections import deque

class ChatMemory:
    def __init__(self, max_turns: int = 5):
        self.store: Dict[str, deque] = {}
        self.max_messages = max_turns * 2

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.store:
            self.store[session_id] = deque(maxlen=self.max_messages)
        self.store[session_id].append({"role": role, "parts": [{"text": content}]})

    def get_history(self, session_id: str) -> List[Dict]:
        return list(self.store.get(session_id, []))

    def clear_history(self, session_id: str):
        if session_id in self.store:
            del self.store[session_id]

memory = ChatMemory()