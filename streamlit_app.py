import uuid

import streamlit as st

from app import run_agents


# ======================
# Настройка страницы
# ======================
st.set_page_config(
    page_title="Цифровой ассистент Синергия",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================
# Кастомные стили CSS
# ======================
st.markdown("""
<style>
    /* Основные стили */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        margin-top: 0.5rem;
        font-size: 1.1rem;
    }
    
    /* Карточки агентов */
    .agent-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .swot-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    /* Кнопки */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4);
    }
    
    /* Боковая панель */
    .sidebar-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    
    /* Прогресс-бар */
    .progress-container {
        margin: 1rem 0;
    }
    
    /* Улучшенные текстовые блоки */
    .result-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    
    /* Иконки для SWOT */
    .swot-section {
        margin: 1.5rem 0;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .strengths { background: #d4edda; border-left: 4px solid #28a745; }
    .weaknesses { background: #fff3cd; border-left: 4px solid #ffc107; }
    .opportunities { background: #d1ecf1; border-left: 4px solid #17a2b8; }
    .threats { background: #f8d7da; border-left: 4px solid #dc3545; }
    
    /* Карточки вариантов */
    .option-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .option-card h4 {
        margin-top: 0;
        color: #667eea;
    }
    
    /* Рейтинг */
    .rating-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# Заголовок
# ======================
st.markdown("""
<div class="main-header">
    <h1>Цифровой ассистент руководства компании Синергия</h1>
    <p>Многоагентная система анализа: Future → Web Search → RAG → SWOT-ранжирование</p>
</div>
""", unsafe_allow_html=True)

# ======================
# Боковая панель
# ======================
with st.sidebar:
    st.markdown("### 📊 Информация о системе")
    st.markdown("""
    <div class="sidebar-info">
        <p><strong>🔄 Процесс работы:</strong></p>
        <ol>
            <li><strong>Future-агент</strong> — анализ будущих перспектив и трендов</li>
            <li><strong>WebSearch-агент</strong> — поиск актуальной информации в интернете</li>
            <li><strong>RAG-агент</strong> — поиск в корпоративных документах</li>
            <li><strong>SWOT-анализ</strong> — ранжирование всех вариантов решения</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    if "session_id" in st.session_state:
        st.markdown("---")
        st.markdown("### 🔑 Сессия")
        st.code(st.session_state.session_id[:8] + "...", language=None)

# ======================
# Инициализация состояния
# ======================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "show_rag" not in st.session_state:
    st.session_state.show_rag = False

if "show_web" not in st.session_state:
    st.session_state.show_web = False

if "strategy_swot_visible" not in st.session_state:
    st.session_state.strategy_swot_visible = {}

# ======================
# Форма запроса
# ======================
st.markdown("### 💬 Введите ваш запрос")

with st.form("user_query_form", clear_on_submit=False):
    user_query = st.text_area(
        "Опишите задачу или вопрос",
        height=120,
        placeholder="Например: Проанализируй текущую ситуацию с развитием IT-направления в компании..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        submitted = st.form_submit_button("🚀 Запустить анализ", use_container_width=True)
    with col2:
        if st.form_submit_button("🗑️ Очистить", use_container_width=True):
            st.session_state.last_result = None
            st.session_state.show_rag = False
            st.session_state.show_web = False
            st.session_state.strategy_swot_visible = {}
            st.rerun()

# ======================
# Обработка запроса
# ======================
if submitted:
    if not user_query.strip():
        st.warning("⚠️ Пожалуйста, введите запрос перед запуском анализа.")
    else:
        # Прогресс-индикаторы
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.info("🔄 **Анализ в процессе**")
        progress_bar.progress(25)
        
        try:
            result = run_agents(st.session_state.session_id, user_query.strip())
            st.session_state.last_result = result
            st.session_state.strategy_swot_visible = {}
            
            progress_bar.progress(100)
            status_text.success("✅ **Анализ завершён!** Результаты готовы к просмотру.")
            progress_bar.empty()
            
            # Небольшая задержка для визуального эффекта
            import time
            time.sleep(0.5)
            st.rerun()
            
        except Exception as e:
            progress_bar.empty()
            status_text.error(f"❌ **Ошибка:** {str(e)}")
            st.exception(e)

# ======================
# Отображение результатов
# ======================
result = st.session_state.last_result

if result:
    st.markdown("---")
    
    # Вкладки для разных представлений
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Future-агент", 
        "🌐 WebSearch-агент", 
        "📚 RAG-агент", 
        "🎯 Стратегии"
    ])
    
    # ======================
    # Вкладка 1: Future агент
    # ======================
    with tab1:
        st.markdown("### 🔮 Анализ будущих перспектив")
        st.markdown("Варианты решения, ориентированные на будущее развитие (1-3 года)")
        
        st.markdown(f'<div class="result-box">{result.future_answer}</div>', unsafe_allow_html=True)
    
    # ======================
    # Вкладка 2: Web Search агент
    # ======================
    with tab2:
        st.markdown("### 🌐 Что делают другие компании")
        st.markdown("Анализ подходов и практик других игроков на рынке")
        
        st.markdown(f'<div class="result-box">{result.web_answer}</div>', unsafe_allow_html=True)
    
    # ======================
    # Вкладка 3: RAG агент
    # ======================
    with tab3:
        st.markdown("### 📚 Анализ внутренней информации")
        st.markdown("Анализ внутренних документов, встреч и решений компании")
        
        st.markdown(f'<div class="result-box">{result.rag_answer}</div>', unsafe_allow_html=True)
    
    # ======================
    # Вкладка 4: Стратегии и SWOT
    # ======================
    with tab4:
        st.markdown("### 🎯 Итоговые стратегии и ранжирование")
        
        combined_summary = getattr(result, "combined_summary", "")
        if combined_summary:
            st.markdown("#### 💡 Краткий вывод")
            st.info(combined_summary)
        
        raw_strategies = getattr(result, "strategies", []) or []
        if not raw_strategies:
            st.warning("Стратегии пока не сформированы.")
        else:
            strategies = []
            for strat in raw_strategies:
                if hasattr(strat, "dict"):
                    strategies.append(strat.dict())
                else:
                    strategies.append(strat)
            strategies.sort(key=lambda x: x.get("rank", 999))
            
            for strat in strategies:
                name = strat.get("name") or "Стратегия"
                description = strat.get("description") or ""
                rationale = strat.get("rationale") or ""
                rank = strat.get("rank") or 0
                swot = strat.get("swot") or {}
                
                st.markdown(f"""
                <div class="option-card">
                    <h4>🏆 Ранг #{rank}: {name}</h4>
                    <p><strong>Кратко:</strong> {description}</p>
                    <p><strong>Почему важно:</strong> {rationale}</p>
                </div>
                """, unsafe_allow_html=True)
                
                toggle_key = f"swot_visibility_{rank}"
                current_state = st.session_state.strategy_swot_visible.get(toggle_key, False)
                button_label = "Показать SWOT" if not current_state else "Скрыть SWOT"
                if st.button(button_label, key=f"swot_btn_{rank}", use_container_width=True):
                    current_state = not current_state
                    st.session_state.strategy_swot_visible[toggle_key] = current_state
                
                if st.session_state.strategy_swot_visible.get(toggle_key, False):
                    strengths = swot.get("strengths", [])
                    weaknesses = swot.get("weaknesses", [])
                    opportunities = swot.get("opportunities", [])
                    threats = swot.get("threats", [])
                    
                    sections = [
                        ("Сильные стороны (Strengths)", strengths),
                        ("Слабые стороны (Weaknesses)", weaknesses),
                        ("Возможности (Opportunities)", opportunities),
                        ("Угрозы (Threats)", threats),
                    ]
                    for title, items in sections:
                        if items:
                            st.markdown(f"**{title}:**")
                            for item in items:
                                st.markdown(f"- {item}")
                        else:
                            st.markdown(f"**{title}:** —")
                    st.markdown("---")

else:
    # Приветственное сообщение
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px;">
            <h3>👋 Добро пожаловать!</h3>
            <p style="font-size: 1.1rem; color: #666;">
                Введите ваш запрос выше и нажмите <strong>"🚀 Запустить анализ"</strong><br>
                Система проанализирует информацию из трёх источников и предоставит SWOT-анализ с ранжированием вариантов
            </p>
        </div>
        """, unsafe_allow_html=True)
