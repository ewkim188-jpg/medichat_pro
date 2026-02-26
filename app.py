import streamlit as st
from model import qa_bot

# -----------------------------------------------------------------------------
# 1. 페이지 설정 (반드시 가장 먼저 호출)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MediChat Pro: Medical AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. 커스텀 CSS 스타일링 (Medical Theme)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* 전체 페이지 배경 */
    .stApp {
        background-color: #f1f5f9; /* Slate 100 */
        font-family: 'Inter', sans-serif;
    }

    /* 사이드바 스타일 (Dark Blue/Navy) */
    [data-testid="stSidebar"] {
        background-color: #0f172a; /* Slate 900 */
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important; /* Slate 200 */
    }
    [data-testid="stSidebar"] .stButton > button {
        background-color: #334155; /* Slate 700 */
        color: white !important;
        border: 1px solid #475569;
    }

    /* 헤더 영역 스타일 */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 1rem;
        border-bottom: 1px solid #cbd5e1;
        margin-bottom: 2rem;
        color: #64748b;
    }

    /* 채팅 메시지 카드 스타일 */
    .stChatMessage {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    /* 아바타 스타일 */
    .stChatMessage .stChatMessageAvatar {
        background-color: #3b82f6; /* Blue 500 */
    }

    /* Disclaimer 박스 스타일 */
    .disclaimer-box {
        background-color: #fff7ed; /* Orange 50 */
        border-left: 4px solid #f97316; /* Orange 500 */
        padding: 1rem;
        border-radius: 6px;
        color: #9a3412;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }

    /* 타이틀 스타일 */
    h1 {
        color: #1e293b; /* Slate 800 */
        font-weight: 700;
        letter-spacing: -0.025em;
    }
    
    /* Expander 스타일 */
    .streamlit-expanderHeader {
        background-color: white;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. 세션 상태 초기화
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------------------------------------------------
# 4. 사이드바 구성 (설정 및 정보)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063822.png", width=80) 
    st.title("MediChat Pro")
    st.caption("v1.0.0 | Powered by Llama-2 & RAG")
    
    st.markdown("---")
    
    # 페르소나 선택
    st.subheader("👤 모드 선택 (Persona)")
    persona = st.radio(
        "답변 스타일을 선택하세요:",
        ("환자 (Patient) 🙆", "의사 (Doctor) 👨‍⚕️"),
        index=0
    )
    mode = "Patient" if "환자" in persona else "Doctor"
    
    st.info(f"현재 **{mode} 모드**로 대화 중입니다.")
    
    st.markdown("---")

    # 대화 초기화 버튼
    if st.button("🗑️ 대화 내용 지우기 (Clear Chat)"):
        st.session_state.messages = []
        st.experimental_rerun()

    # 면책 조항 (Disclaimer) - 제약사 필수
    st.markdown("---")
    st.warning("⚠️ **Disclaimer**")
    st.markdown("""
    <div style='font-size: 0.8em; color: #555;'>
    본 서비스는 AI 기반 정보 제공 시스템으로, 
    의사의 전문적인 진료를 대신할 수 없습니다. 
    정확한 진단과 처방은 반드시 전문의와 상담하십시오.
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. 메인 화면 구성
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 5. 메인 화면 구성
# -----------------------------------------------------------------------------

# 대시보드 헤더 (Breadcrumb Style)
st.markdown("""
    <div class="header-container">
        <div style="font-size: 1.2rem; font-weight: 600; color: #334155;">
            Dashboard &nbsp; <span style="color: #cbd5e1;">/</span> &nbsp; <span style="color: #64748b;">Medical Chat Analysis</span>
        </div>
        <div style="font-size: 0.85rem; color: #94a3b8;">
            Last analysis: 2026-02-20 12:30:00
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("<h1>Medical Chatbot <span style='color:#3b82f6; font-size:1.5rem; vertical-align:middle;'>Pro</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#64748b; margin-top:-15px; margin-bottom: 30px;'>제약 서비스를 위한 이중 페르소나 AI 시스템 (Dual Persona AI System)</p>", unsafe_allow_html=True)

# 상단 Disclaimer 표시
st.markdown("""
<div class="disclaimer-box">
    <b>[안내]</b> 이 챗봇은 의학 논문 및 가이드라인에 기반하여 답변합니다. 
    <b>환자 모드</b>에서는 쉬운 용어로, <b>의사 모드</b>에서는 전문 용어로 설명합니다.
</div>
""", unsafe_allow_html=True)

# 대화 기록 표시
for message in st.session_state.messages:
    if message["role"] == "user":
        avatar = "👤"
    else:
        avatar = "🤖" if mode == "Doctor" else "💊"
        
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# -----------------------------------------------------------------------------
# 6. 채팅 입력 및 처리 로직
# -----------------------------------------------------------------------------

# 리소스 캐싱 함수 (모델 로딩 시간 단축 및 메모리 에러 방지)
@st.cache_resource
def get_qa_bot():
    return qa_bot()

if prompt := st.chat_input("질문을 입력하세요... (예: 고혈압 약 부작용이 뭐야?)"):
    # 1. 사용자 메시지 표시
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. AI 답변 생성
    try:
        with st.chat_message("assistant", avatar="🤖" if mode == "Doctor" else "💊"):
            message_placeholder = st.empty()
            
            with st.spinner(f"AI가 {mode} 관점에서 분석 중입니다..."):
                # 챗봇 호출 (캐시된 버전 사용)
                qa = get_qa_bot()
                res = qa({"query": prompt, "mode": mode}) 
                answer = res["result"]
                sources = res["source_documents"]
                
                # 답변 표시
                message_placeholder.markdown(answer)
                
                # 출처(Reference) 표시 - Expander로 깔끔하게 정리
                if sources:
                    with st.expander("📚 참고 문헌 확인 (Debug: Retrieved Sources)"):
                        for i, doc in enumerate(sources):
                            source_name = doc.metadata.get('source', 'Unknown')
                            page_num = doc.metadata.get('page', 'N/A')
                            st.markdown(f"**{i+1}. {source_name}**")
                            st.text(doc.page_content[:300]) # 내용 미리보기 확대

        # 3. 대화 기록 저장
        st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"시스템 오류가 발생했습니다: {str(e)}")
        st.info("관리자에게 문의해주세요.")
