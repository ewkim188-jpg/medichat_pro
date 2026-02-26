# """ streamlit run medical_chatbot/app.py """
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
from peft import PeftModel, PeftConfig
import os

DB_FAISS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectorstore/db_faiss")

# 듀얼 페르소나 프롬프트 (Dual Persona Prompts) - Korean Version

# 1. 의사 모드 (Doctor Mode): 전문성, 근거 중심, 간결함 (한국어)
# 듀얼 페르소나 프롬프트 (Dual Persona Prompts) - Korean Version (Llama-2 Chat Format)

# 1. 의사 모드 (Doctor Mode): 전문성, 근거 중심, 간결함 (한국어)
doctor_prompt_template = """[INST] <<SYS>>
You are an expert AI medical consultant for a pharmaceutical company.
- You MUST answer in **Korean (한국어)** only.
- Do NOT use English. content must be fully translated.
- When asked about new drugs or clinical trials, explain the mechanism of action (MOA) and recent clinical trial results (Advance-HTN, Launch-HTN) clearly.
- If the answer is not in the context, say "현재 제공된 데이터에는 해당 신약에 대한 정보가 없습니다." in Korean.
<</SYS>>

Context: {context}
Question: {question}

Helpful Answer (in detailed Korean): [/INST]"""

# 2. 환자 모드 (Patient Mode): 공감, 쉬운 설명, 안심 (한국어)
patient_prompt_template = """[INST] <<SYS>>
You are an AI health assistant for patients.
- You MUST answer in **Korean (한국어)** only.
- Do NOT use English. Explain in simple Korean.
- If the answer is not in the context, say "죄송합니다. 현재 정보로는 답변드리기 어렵습니다. 담당 의사와 상담해주세요." in Korean.
<</SYS>>

Context: {context}
Question: {question}

Helpful Answer (in friendly Korean): [/INST]"""

def get_prompt_template(mode):
    if mode == "Doctor":
        return doctor_prompt_template
    else:
        return patient_prompt_template

def load_llm():
    # Base Model (QLoRA)
    base_model_name = "NousResearch/Llama-2-7b-chat-hf"
    # 절대 경로로 어댑터 모델 경로 설정
    adapter_model_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama-2-7b-med-chatbot")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Load Adapter (Fine-Tuned weights)
        model = PeftModel.from_pretrained(model, adapter_model_name)
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Create a pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15,
            return_full_text=False  # 프롬프트가 답변에 포함되지 않도록 설정
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        return llm
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

class ManualRAG:
    def __init__(self, llm, db):
        self.llm = llm
        self.db = db

    def __call__(self, inputs):
        """
        RAG 파이프라인 실행
        inputs: {'query': '질문 내용', 'mode': 'Patient' or 'Doctor'}
        """
        query = inputs['query']
        mode = inputs.get('mode', 'Patient') # 기본값은 환자 모드
        
        # 1. 검색 (Retrieve)
        # 유사도가 높은 상위 2개 문서 검색
        docs = self.db.similarity_search(query, k=2)
        
        # 2. 문맥 생성 (Context)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 3. 프롬프트 선택 및 구성 (Augment)
        prompt_template = get_prompt_template(mode)
        prompt = prompt_template.format(context=context, question=query)
        
        # 4. 생성 (Generate)
        response_text = self.llm.invoke(prompt) # invoke 또는 직접 호출
        
        return {
            "result": response_text,
            "source_documents": docs
        }

def qa_bot():
    """
    챗봇 인스턴스를 초기화하고 반환하는 메인 함수
    """
    # 임베딩 모델 초기화
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    
    # 로컬에 저장된 벡터 DB 로드
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    llm = load_llm() # LLM 로드
    if llm is None:
        raise ValueError("Could not load LLM. Check model path or connection.")
        
    # 수동 RAG 클래스 반환
    qa = ManualRAG(llm, db)

    return qa

def final_result(query, mode="Patient"):
    """
    사용자 질문을 받아 답변을 반환하는 최종 인터페이스 함수
    """
    qa_result = qa_bot() # 챗봇 생성
    response = qa_result({'query': query, 'mode': mode}) # 질문 실행
    return response # 결과 반환
