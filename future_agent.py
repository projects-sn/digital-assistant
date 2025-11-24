# future_agent.py
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

# =========================
# Конфиг
# =========================
FUTURE_MODEL = os.getenv("FUTURE_MODEL", "gpt-4o")

# =========================
# Память диалогов
# =========================
class SessionStore:
    """Хранение истории сообщений по session_id."""
    def __init__(self) -> None:
        self._store: Dict[str, InMemoryChatMessageHistory] = {}

    def get_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self._store:
            self._store[session_id] = InMemoryChatMessageHistory()
        return self._store[session_id]

SESSION_STORE = SessionStore()

# =========================
# Промпт для анализа будущих перспектив
# =========================
future_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Ты — стратегический аналитик компании Синергия, специализирующийся на анализе будущих перспектив и трендов. "
     "Твоя задача — предложить 3 варианта решения/подхода, ориентированных на будущее развитие (1-3 года). "
     "Каждый вариант должен быть:\n"
     "- Конкретным и реализуемым\n"
     "- Ориентированным на долгосрочную перспективу\n"
     "- Учитывающим современные тренды и технологии\n"
     "- С обоснованием перспективности\n\n"
     "Формат ответа — структурированный текст с тремя вариантами, каждый из которых содержит:\n"
     "- Название варианта\n"
     "- Описание подхода/решения\n"
     "- Обоснование перспективности\n"
     "- Ключевые действия для реализации\n"
     "- Ориентировочные сроки\n\n"
     "Варианты должны быть ранжированы по приоритету (первый — наиболее перспективный)."),
    MessagesPlaceholder("history"),
    ("user", "{user_query}")
])

# =========================
# Цепочка для Future агента
# =========================
future_chain_base = future_prompt | ChatOpenAI(model=FUTURE_MODEL, temperature=0.7) | StrOutputParser()

def _get_history(session_id: str) -> InMemoryChatMessageHistory:
    return SESSION_STORE.get_history(session_id)

future_chain = RunnableWithMessageHistory(
    future_chain_base,
    lambda session_id: _get_history(session_id),
    input_messages_key="user_query",
    history_messages_key="history",
)

# =========================
# Внешний интерфейс
# =========================
@dataclass
class FutureResult:
    session_id: str
    answer_text: str
    options: List[Dict[str, Any]] = field(default_factory=list)
    raw: str = ""

def chat(session_id: str, user_query: str) -> FutureResult:
    """
    Основной вход: на входе session_id и текст пользователя.
    Возвращает анализ будущих перспектив с вариантами решения.
    """
    config = {"configurable": {"session_id": session_id}}
    result_text = future_chain.invoke({"user_query": user_query}, config=config)
    
    # Попытка извлечь структурированные варианты из текста
    options = _extract_options(result_text)
    
    return FutureResult(
        session_id=session_id,
        answer_text=result_text,
        options=options,
        raw=result_text
    )

def _extract_options(text: str) -> List[Dict[str, Any]]:
    """
    Пытается извлечь варианты из текста ответа.
    Ищет паттерны типа "Вариант 1:", "Вариант 2:", "Вариант 3:" и т.д.
    """
    import re
    options = []
    
    # Паттерн для поиска вариантов
    pattern = r'(?:Вариант|Вариант решения|Подход|Решение)\s*(\d+)[:\-]?\s*(.*?)(?=(?:Вариант|Вариант решения|Подход|Решение)\s*\d+|$)'
    matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
    
    for match in matches:
        variant_num = match.group(1)
        variant_text = match.group(2).strip()
        
        # Извлечение названия
        title_match = re.search(r'^(.*?)(?:\n|$)', variant_text)
        title = title_match.group(1).strip() if title_match else f"Вариант {variant_num}"
        
        # Извлечение описания
        desc_match = re.search(r'(?:Описание|Суть)[:\-]?\s*(.*?)(?=(?:Обоснование|Ключевые действия|Сроки)|$)', variant_text, re.IGNORECASE | re.DOTALL)
        description = desc_match.group(1).strip() if desc_match else variant_text[:300]
        
        # Извлечение обоснования
        rationale_match = re.search(r'(?:Обоснование|Почему|Перспективность)[:\-]?\s*(.*?)(?=(?:Ключевые действия|Сроки|Вариант)|$)', variant_text, re.IGNORECASE | re.DOTALL)
        rationale = rationale_match.group(1).strip() if rationale_match else ""
        
        # Извлечение ключевых действий
        actions_match = re.search(r'(?:Ключевые действия|Действия|Шаги)[:\-]?\s*(.*?)(?=(?:Сроки|Вариант)|$)', variant_text, re.IGNORECASE | re.DOTALL)
        actions_text = actions_match.group(1).strip() if actions_match else ""
        key_actions = [a.strip() for a in re.split(r'[•\-\n]', actions_text) if a.strip()][:5]
        
        # Извлечение сроков
        timeline_match = re.search(r'(?:Сроки|Временные рамки|Ориентировочно)[:\-]?\s*(.*?)(?=(?:Вариант|$))', variant_text, re.IGNORECASE | re.DOTALL)
        timeline = timeline_match.group(1).strip() if timeline_match else "1-3 года"
        
        options.append({
            "title": title,
            "description": description,
            "rationale": rationale,
            "key_actions": key_actions,
            "timeline": timeline
        })
    
    # Если не нашли структурированные варианты, разобьём текст на части
    if not options:
        parts = re.split(r'\n\n+', text)
        for i, part in enumerate(parts[:3], 1):
            if len(part.strip()) > 50:
                options.append({
                    "title": f"Вариант {i}",
                    "description": part.strip()[:500],
                    "rationale": "",
                    "key_actions": [],
                    "timeline": "1-3 года"
                })
    
    return options[:3]  # Максимум 3 варианта

