# app.py
import os
import re
import uuid
from pydantic import BaseModel
import json
from typing import Optional, Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

# Импортируем агентов
import rag_agent
import web_search_agent
import future_agent



# Инициализация Summarizer агента

SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-4o")

strategy_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Ты — стратегический аналитик Синергии. На основе ответов трёх агентов (Future, WebSearch, RAG) "
     "нужно предложить и оценить стратегии действий компании. Сначала объединяешь ключевые выводы, затем формируешь "
     "ровно три стратегии (Стратегия 1/2/3), каждая из которых должна:\n"
     "- отражать, что стоит взять на заметку компании\n"
     "- содержать краткое описание сути стратегии\n"
     "- объяснять, почему эта стратегия важна (обоснование)\n"
     "- включать SWOT-блок по этой стратегии (S/W/O/T)\n"
     "- иметь оценку приоритета (rank от 1 до 3, где 1 лучший)\n"
     "Верни строго корректный JSON со следующими полями:\n"
     "- summary: строка с кратким выводом (2-3 предложения)\n"
     "- strategies: массив из трёх объектов. Каждый объект содержит\n"
     "  - name: строка с названием стратегии\n"
     "  - description: краткое описание сути\n"
     "  - rationale: почему это важно\n"
     "  - rank: число от 1 до 3 (1 — лучший)\n"
     "  - swot: объект со списками strengths/weaknesses/opportunities/threats (массивы строк)\n"
     "Не добавляй текст вне JSON. Стратегии ранжируй по возрастанию rank."),
    MessagesPlaceholder("history"),
    ("user",
     "Пользовательский запрос:\n{user_query}\n\n"
     "Ответ Future агента:\n{future_answer}\n\n"
     "Ответ WebSearch агента:\n{web_answer}\n\n"
     "Ответ RAG агента:\n{rag_answer}\n\n"
     "Сформируй JSON с summary и тремя стратегиями с SWOT и rank.")
])

strategy_chain = strategy_prompt | ChatOpenAI(model=SUMMARY_MODEL, temperature=0.3) | StrOutputParser()


def _strip_code_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    return cleaned.strip()


def _ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace("•", "\n").splitlines() if p.strip()]
        if parts:
            return parts
    return [str(value).strip()]


def _parse_strategy_output(text: str) -> Dict[str, Any]:
    cleaned = _strip_code_fence(text)
    data: Dict[str, Any] = {"summary": cleaned or "", "strategies": []}
    if not cleaned:
        return data

    try:
        parsed = json.loads(cleaned)
    except Exception:
        return data

    summary = (parsed.get("summary") or "").strip()
    strategies = []
    for idx, raw in enumerate(parsed.get("strategies") or [], 1):
        swot_raw = raw.get("swot") or {}
        try:
            rank_val = int(raw.get("rank", idx))
        except (TypeError, ValueError):
            rank_val = idx
        strategies.append({
            "name": (raw.get("name") or f"Стратегия {idx}").strip(),
            "description": (raw.get("description") or raw.get("summary") or "").strip(),
            "rationale": (raw.get("rationale") or raw.get("why") or "").strip(),
            "rank": rank_val,
            "swot": {
                "strengths": _ensure_list(swot_raw.get("strengths")),
                "weaknesses": _ensure_list(swot_raw.get("weaknesses")),
                "opportunities": _ensure_list(swot_raw.get("opportunities")),
                "threats": _ensure_list(swot_raw.get("threats")),
            },
        })

    strategies.sort(key=lambda x: x.get("rank", 999))
    return {
        "summary": summary or data["summary"],
        "strategies": strategies[:3],
    }


# Память сообщений

class SessionStore:
    def __init__(self):
        self.store: Dict[str, InMemoryChatMessageHistory] = {}

    def get(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

SESSION_STORE = SessionStore()

strategizer_chain = RunnableWithMessageHistory(
    strategy_chain,
    lambda session_id: SESSION_STORE.get(session_id),
    input_messages_key="user_query",
    history_messages_key="history",
)



class StrategyItem(BaseModel):
    name: str
    description: str
    rationale: str
    rank: int
    swot: Dict[str, List[str]]


class ChatResponse(BaseModel):
    session_id: str
    future_answer: str
    web_answer: str
    rag_answer: str
    combined_summary: str
    strategies: List[StrategyItem]


# Основная функция общения

def run_agents(session_id: str, user_message: str) -> ChatResponse:
    # 1. Запуск Future агента (анализ будущих перспектив, 3 варианта)
    future_res = future_agent.chat(session_id, user_message)
    future_answer = future_res.answer_text

    # 2. Запуск Web Search агента (что делают другие компании)
    web_res = web_search_agent.chat(session_id, user_message)
    web_answer = web_res.answer_text

    # 3. Запуск RAG агента (что делала сама компания)
    rag_chain = rag_agent.build_rag_chain()
    rag_result = rag_chain({"query": user_message})
    rag_answer = rag_result["result"]

    # 4. Суммаризация и построение стратегий + SWOT
    config = {"configurable": {"session_id": session_id}}
    strategy_raw = strategizer_chain.invoke(
        {
            "user_query": user_message,
            "future_answer": future_answer,
            "web_answer": web_answer,
            "rag_answer": rag_answer,
        },
        config=config,
    )
    parsed = _parse_strategy_output(strategy_raw)
    combined_summary = parsed.get("summary", "")
    strategy_items: List[StrategyItem] = []
    for item in parsed.get("strategies", []):
        try:
            strategy_items.append(StrategyItem(**item))
        except Exception:
            continue

    return ChatResponse(
        session_id=session_id,
        future_answer=future_answer,
        web_answer=web_answer,
        rag_answer=rag_answer,
        combined_summary=combined_summary,
        strategies=strategy_items,
    )


# Локальный запуск CLI (для ручного тестирования)
if __name__ == "__main__":
    session = str(uuid.uuid4())
    print("Старт локального CLI. Введите вопрос (пустая строка для выхода).")
    while True:
        try:
            user_text = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_text:
            break
        response = run_agents(session, user_text)
        print("\n--- Итоговый вывод ---")
        print(response.combined_summary or "<нет данных>")
        if response.strategies:
            print("\n--- Стратегии ---")
            for strat in response.strategies:
                print(f"[#{strat.rank}] {strat.name}: {strat.description}")
        print("\n")
