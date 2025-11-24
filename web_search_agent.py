import os, json, time, uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from fastapi import FastAPI
from pydantic import BaseModel, Field

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv
load_dotenv()


# OpenAI (Responses API для web_search)
from openai import OpenAI

# =========================
# Конфиг
# =========================
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
REPHRASE_MODEL = os.getenv("REPHRASE_MODEL", OPENAI_MODEL)
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", OPENAI_MODEL)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# Память диалогов (in-memory)
# =========================
class SessionStore:
    """Простое хранение истории сообщений по session_id. При желании замените на Redis."""
    def __init__(self) -> None:
        self._store: Dict[str, InMemoryChatMessageHistory] = {}

    def get_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self._store:
            self._store[session_id] = InMemoryChatMessageHistory()
        return self._store[session_id]

SESSION_STORE = SessionStore()

# =========================
# 1) Перефраз под аналоги/кейсы в РФ (LangChain LLM)
# =========================
rephrase_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Ты помощник-ресерчер. Перефразируй пользовательский запрос в короткий поисковый "
     "запрос на русском языке, ориентированный на аналоги/кейсы/рынок в России. "
     "Добавляй при необходимости: 'в России', 'кейсы', 'аналог', 'практика', 'рынок РФ'. "
     "Верни только одну строку, 5–15 слов, без комментариев."),
    MessagesPlaceholder("history"),
    ("user", "{user_query}")
])

rephraser = rephrase_prompt | ChatOpenAI(model=REPHRASE_MODEL, temperature=0.2) | StrOutputParser()

# =========================
# 2) Вызов встроенного OpenAI web_search (Responses API)
#    Возвращаем строго структурированный JSON: { rewritten, bullets[], sources[] }
# =========================
def _call_openai_web_search(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    inputs: {
      "orig_query": str,
      "rewritten": str
    }
    Returns:
      {
        "rewritten": str,
        "bullets": [str, ...],
        "summary": str,
        "sources": [{"title": str, "url": str}]
      }
    """
    orig = inputs["orig_query"]
    rewritten = inputs["rewritten"]

    SYSTEM = (
        "Ты исследователь. Используй инструмент web_search, чтобы найти свежие данные. "
        "Цель: показать аналоги/кейсы/игроков и тренды ИМЕННО в России. "
        "Формат ответа — JSON со следующими полями:\n"
        "{\n"
        '  "rewritten": "<строка перефраза>",\n'
        '  "summary": "<4-6 предложений краткого обзора>",\n'
        '  "bullets": ["краткий факт 1", "краткий факт 2", "..."],\n'
        '  "sources": [{"title": "<заголовок>", "url": "<ссылка>"}]\n'
        "}\n"
        "Источники должны соответствовать найденным страницам. Не выдумывай ссылки."
    )

    # Запрашиваем у Responses API с включенным web_search
    resp = client.responses.create(
        model=SUMMARY_MODEL,
        tools=[{"type": "web_search"}],
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM}]},
            {"role": "user", "content": [
                {"type": "input_text",
                 "text": f"Исходный запрос: {orig}\nПерефраз для поиска: {rewritten}\n"
                         f"Сделай web_search и верни JSON."}
            ]},
        ],
    )

    # Достаём финальный текст (должен быть JSON); параллельно собираем url_citation
    final_text = ""
    citations: List[Dict[str, str]] = []
    for item in resp.output or []:
        if item.type == "message":
            for c in (item.content or []):
                if getattr(c, "type", "") == "output_text":
                    final_text += c.text or ""
                    for ann in (getattr(c, "annotations", []) or []):
                        if getattr(ann, "type", "") == "url_citation":
                            citations.append({"title": ann.title, "url": ann.url})

    # Пытаемся распарсить JSON, если модель уже вернула JSON
    parsed: Dict[str, Any] = {}
    try:
        parsed = json.loads(final_text)
    except Exception:
        # Если вернуло не-JSON, упакуем как summary с источниками из аннотаций
        parsed = {
            "rewritten": rewritten,
            "summary": final_text.strip(),
            "bullets": [],
            "sources": citations
        }

    # Если внутри JSON нет sources — подставим из аннотаций
    if isinstance(parsed, dict) and not parsed.get("sources") and citations:
        parsed["sources"] = citations

    # Гарантируем минимальные поля
    parsed.setdefault("rewritten", rewritten)
    parsed.setdefault("summary", "")
    parsed.setdefault("bullets", [])
    parsed.setdefault("sources", [])
    return parsed

web_search_runnable = RunnableLambda(_call_openai_web_search)

# =========================
# 3) Главный граф: rephrase -> web_search -> сбор ответа
#    + Message History (RunnableWithMessageHistory)
# =========================
def _format_answer(data: Dict[str, Any]) -> str:
    """Формирует читабельный ответ для клиента + список источников."""
    parts = []
    if data.get("summary"):
        parts.append(data["summary"])
    bullets = data.get("bullets") or []
    if bullets:
        parts.append("\n— " + "\n— ".join(bullets))
    sources = data.get("sources") or []
    if sources:
        src_lines = []
        for i, s in enumerate(sources, 1):
            t = s.get("title") or "Источник"
            u = s.get("url") or ""
            src_lines.append(f"[{i}] {t} — {u}")
        parts.append("\nИсточники:\n" + "\n".join(src_lines))
    return "\n".join([p for p in parts if p]).strip()

def _chain_builder() -> RunnableParallel:
    # Параллельно запускаем rephraser и пробрасываем вход в web_search
    # Затем web_search использует rewritten.
    return (
        RunnableParallel(
            rewritten=rephraser,
            orig_query=RunnableLambda(lambda x: x["user_query"])
        )
        | RunnableLambda(lambda x: {"orig_query": x["orig_query"], "rewritten": x["rewritten"]})
        | web_search_runnable
    )

base_chain = _chain_builder()

def _get_history(session_id: str) -> InMemoryChatMessageHistory:
    return SESSION_STORE.get_history(session_id)

# Оборачиваем chain в память сообщений
chat_chain = RunnableWithMessageHistory(
    base_chain,
    lambda session_id: _get_history(session_id),
    input_messages_key="user_query",
    history_messages_key="history",
)

# =========================
# 4) Внешний интерфейс
# =========================
@dataclass
class ChatResult:
    session_id: str
    rewritten: str
    answer_text: str
    sources: List[Dict[str, str]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

def chat(session_id: str, user_query: str) -> ChatResult:
    """
    Основной вход: на входе session_id и текст пользователя.
    Возвращает перефраз, ответ и источники.
    """
    config = {"configurable": {"session_id": session_id}}
    result: Dict[str, Any] = chat_chain.invoke({"user_query": user_query}, config=config)
    # result — это JSON от _call_openai_web_search
    answer_text = _format_answer(result)
    return ChatResult(
        session_id=session_id,
        rewritten=result.get("rewritten", ""),
        answer_text=answer_text,
        sources=result.get("sources", []),
        raw=result
    )

# =========================
# 5) FastAPI
# =========================
app = FastAPI(title="Russia-Analogs-Search Bot", version="1.0.0")

class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Уникальный ID сессии (если нет — будет создан)")
    user_message: str

class ChatResponse(BaseModel):
    session_id: str
    rewritten_query: str
    answer: str
    sources: List[Dict[str, str]]

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    sid = req.session_id or str(uuid.uuid4())
    res = chat(sid, req.user_message)
    return ChatResponse(
        session_id=res.session_id,
        rewritten_query=res.rewritten,
        answer=res.answer_text,
        sources=res.sources
    )

# =========================
# 6) Локальный запуск
# =========================
if __name__ == "__main__":
    # Пример CLI-диалога
    sid = str(uuid.uuid4())
    print(f"Session: {sid}")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        out = chat(sid, q)
        print("\n--- Перефраз ---")
        print(out.rewritten)
        print("\n--- Ответ ---")
        print(out.answer_text)
        print("\n")
