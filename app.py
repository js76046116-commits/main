import streamlit as st
import os
import json
import itertools
from sentence_transformers import CrossEncoder 

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

# ==========================================================
# [0] 페이지 및 경로 설정
# ==========================================================
st.set_page_config(page_title="건설 CM AI 검색 엔진", page_icon="🏗️", layout="wide")

# API 키 설정 (Streamlit Secrets에서 가져옴)
if "GOOGLE_API_KEY" not in os.environ:
    pass 

# 경로 설정
DB_PATH_1 = "./chroma_db_part1"
DB_PATH_2 = "./chroma_db_part2"
JSON_DATA_PATH = "./legal_data_total_vlm.json"

# 전역 변수
RAW_DATA = []

# ==========================================================
# [1] 시스템 로딩 (DB + Hybrid Search + Reranker)
# ==========================================================
class SimpleHybridRetriever:
    def __init__(self, bm25, chroma1, chroma2, raw_data):
        self.bm25 = bm25
        self.chroma1 = chroma1
        self.chroma2 = chroma2
        self.raw_data = raw_data
        
    def invoke(self, query):
        # 1. BM25 & Chroma 검색
        docs_bm25 = self.bm25.invoke(query)
        docs_c1 = self.chroma1.invoke(query)
        docs_c2 = self.chroma2.invoke(query)
        
        # 2. ID -> 원본 텍스트 복원
        real_docs_chroma = []
        for doc in (docs_c1 + docs_c2):
            try:
                idx = int(doc.page_content) 
                original_item = self.raw_data[idx] 
                
                content = original_item.get('content', '').strip()
                source = original_item.get('source', '').strip()
                article = original_item.get('article', '').strip()
                full_text = f"[{source}] {content}"
                
                new_doc = Document(page_content=full_text, metadata={"source": source, "article": article})
                real_docs_chroma.append(new_doc)
            except:
                continue

        # 3. 중복 제거
        combined = []
        seen_ids = set()
        
        for d in itertools.chain(docs_bm25, real_docs_chroma):
            key = d.page_content[:30] # 앞 30글자로 중복 판단
            if key not in seen_ids:
                combined.append(d)
                seen_ids.add(key)
                
        return combined[:200]

@st.cache_resource
def load_search_system():
    global RAW_DATA
    
    # JSON 로드
    if not os.path.exists(JSON_DATA_PATH):
        st.error("❌ JSON 데이터 파일이 없습니다.")
        st.stop()
    with open(JSON_DATA_PATH, 'r', encoding='utf-8') as f:
        RAW_DATA = json.load(f)

    # 임베딩 모델
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Chroma DB 로드
    if not os.path.exists(DB_PATH_1) or not os.path.exists(DB_PATH_2):
        st.error("❌ DB 폴더(part1, part2)가 없습니다.")
        st.stop()

    store1 = Chroma(persist_directory=DB_PATH_1, embedding_function=embeddings, collection_name="construction_laws")
    retriever1 = store1.as_retriever(search_kwargs={"k": 100})

    store2 = Chroma(persist_directory=DB_PATH_2, embedding_function=embeddings, collection_name="construction_laws")
    retriever2 = store2.as_retriever(search_kwargs={"k": 100})

    # BM25 생성
    docs = []
    for item in RAW_DATA:
        content = item.get('content', '').strip()
        source = item.get('source', '').strip()
        if not content: continue
        doc = Document(page_content=f"[{source}] {content}", metadata={"source": source, "article": item.get('article', '')})
        docs.append(doc)
    
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 150

    # 하이브리드 결합
    hybrid_retriever = SimpleHybridRetriever(bm25_retriever, retriever1, retriever2, RAW_DATA)
    
    # [중요] 메모리 절약형 가벼운 모델 (무료 서버용)
    reranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", model_kwargs={"torch_dtype": "auto"})

    return hybrid_retriever, reranker

# 시스템 초기화
with st.spinner("🚀 AI 엔진(Dual DB) 시동 거는 중..."):
    hybrid_retriever, reranker_model = load_search_system()

# LLM 설정
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# ==========================================================
# [2] RAG 체인 설정
# ==========================================================
# 띄어쓰기 교정
spacing_chain = ChatPromptTemplate.from_template("교정된 한국어 문장만 출력(설명X): {question}").pipe(llm).pipe(StrOutputParser())

# HyDE 키워드 확장
hyde_chain = ChatPromptTemplate.from_template("건설 전문 검색 키워드 5개 나열(콤마 구분, 설명X): {question}").pipe(llm).pipe(StrOutputParser())

# 답변 생성 프롬프트
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "건설 기준 엔지니어입니다. [Context]를 보고 답변하세요. 원문 내용을 있는 그대로 인용하는 것을 최우선으로 하십시오. 출처 표기 필수.\n[Context]\n{context}"),
    ("human", "질문: {question}")
])

def retrieve_and_rerank(query):
    # 1. 검색
    initial_docs = hybrid_retriever.invoke(query)
    if not initial_docs: return []
    
    # 2. 리랭킹 (정확도 순 정렬)
    pairs = [[query, doc.page_content] for doc in initial_docs]
    scores = []
    batch_size = 16 
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        batch_scores = reranker_model.predict(batch)
        scores.extend(batch_scores)
        
    scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:50]]

def format_docs(docs):
    return "\n\n".join([f"<출처: {d.metadata.get('source')} / {d.metadata.get('article')}>\n{d.page_content}" for d in docs])

# 최종 체인
rag_chain = (
    {"context": RunnableLambda(retrieve_and_rerank) | format_docs, "question": RunnablePassthrough()}
    | answer_prompt | llm | StrOutputParser()
)

# ==========================================================
# [3] 웹 UI 메인 로직 (버튼 로직 수정됨)
# ==========================================================
st.title("🏗️ 건설 CM 전문 AI")
st.caption("🚀 1차 직구 검색(Direct) 후 → 원하면 HyDE 심층 검색(Expansion)으로 이어집니다.")

# 1. 채팅 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. 화면에 대화 내용 그리기
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. [핵심 수정] 심층 검색 버튼 로직 (채팅창 밖으로 뺌)
# 마지막 메시지가 AI의 '1차 답변'일 때만 버튼을 보여줌
last_msg = st.session_state.messages[-1] if st.session_state.messages else None
if last_msg and last_msg["role"] == "assistant" and "1차 답변" in last_msg["content"] and "2차" not in last_msg["content"]:
    
    with st.expander("🤔 답변이 부족한가요? (여기를 눌러 심층 검색)"):
        if st.button("🚀 HyDE 심층 검색 실행"):
            # 사용자의 직전 질문 가져오기
            prev_question = st.session_state.messages[-2]["content"]
            
            # 로딩바 표시 (Status 컨테이너)
            with st.status("🧠 전문가 모드(HyDE) 가동 중...", expanded=True) as status:
                st.write("🔧 질문 의도 분석 및 확장 중...")
                hyde_keywords = hyde_chain.invoke({"question": prev_question})
                final_query = f"{prev_question} {hyde_keywords}"
                st.write(f"-> 확장된 키워드: `{hyde_keywords}`")
                
                st.write("🚀 정밀 재검색 및 답변 작성 중...")
                response_2 = rag_chain.invoke(final_query)
                status.update(label="✅ 심층 분석 완료!", state="complete", expanded=False)
            
            # 결과 저장 및 표시
            final_res = f"### 🤖 2차 상세 답변 (HyDE)\n**확장된 검색어:** `{hyde_keywords}`\n\n{response_2}"
            st.session_state.messages.append({"role": "assistant", "content": final_res})
            
            # 중요: 화면을 새로고침해서 방금 얻은 답변을 채팅창에 박제
            st.rerun()

# 4. 사용자 질문 입력창 (맨 아래 위치)
if prompt := st.chat_input("질문을 입력하세요..."):
    # 사용자 질문 표시 및 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 답변 생성
    with st.chat_message("assistant"):
        with st.status("🔍 1차 검색 진행 중...", expanded=True) as status:
            corrected_query = spacing_chain.invoke({"question": prompt})
            response_1 = rag_chain.invoke(corrected_query)
            status.update(label="✅ 1차 검색 완료", state="complete", expanded=False)
        
        msg_content = f"### 🤖 1차 답변\n{response_1}"
        st.markdown(msg_content)
        st.session_state.messages.append({"role": "assistant", "content": msg_content})
        
        # 중요: 답변이 달리면 버튼을 띄우기 위해 화면 새로고침
        st.rerun()