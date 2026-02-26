# """ streamlit run medical_chatbot/app.py """
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
from peft import PeftModel, PeftConfig
import os
from deep_translator import GoogleTranslator

DB_FAISS_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectorstore", "db_faiss"))

# 듀얼 페르소나 프롬프트 (Dual Persona Prompts) - Korean Version

# 1. 의사 모드 (Doctor Mode): 전문성, 근거 중심, 간결함 (한국어)
# 듀얼 페르소나 프롬프트 (Dual Persona Prompts) - Korean Version (Llama-2 Chat Format)

# 1. 의사 모드 (Doctor Mode): 전문성, 근거 중심, 간결함 (한국어)
doctor_prompt_template = """[INST] <<SYS>>
You are an expert English-to-Korean medical translator.
Your ONLY task is to read the provided [Context] and generate a 100% pure Korean translation and summary answering the [Question].
Rule 1: NO English words are allowed in your output. You must translate everything.
Rule 2: Never use Chinese Hanja characters.
<</SYS>>

[Context]:
{context}

[Question]: {question}

Provide the answer strictly in 100% pure Korean without any English:
[/INST] 답변: """

# 2. 환자 모드 (Patient Mode): 공감, 쉬운 설명, 안심 (한국어)
patient_prompt_template = """[INST] <<SYS>>
You are an expert English-to-Korean translator specialized in explaining medicine to patients in simple terms.
Your ONLY task is to read the provided [Context] and generate a 100% pure, easy Korean translation answering the [Question].
Rule 1: NO English words are allowed in your output. You must translate everything.
Rule 2: Never use Chinese Hanja characters.
<</SYS>>

[Context]:
{context}

[Question]: {question}

Provide the answer strictly in 100% pure, simple Korean without any English:
[/INST] 답변: """

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
            temperature=0.01, # 거의 0에 가깝게 설정
            top_p=0.95,
            repetition_penalty=1.3,
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
        # 질문에 '부작용'이 포함된 경우 side_effects 데이터 위주로 검색되도록 유도 가능하지만
        # 여기서는 우선 순위 필터링을 로직으로 구현
        docs_with_scores = self.db.similarity_search_with_score(query, k=5)
        
        # 2. 관련성 필터링 (Relevance Filtering)
        # 질문이 부작용에 관한 것이라면 임상 데이터(Lorundrostat 등) 비중을 낮춤
        is_side_effect_query = any(word in query for word in ["부작용", "합병증", "위험", "안전"])
        
        filtered_docs = []
        for doc, score in docs_with_scores:
            source = doc.metadata.get('source', '')
            # 부작용 질문 시 임상 데이터는 철저히 배제 (특히 영어 성격이 강한 lorundrostat)
            if is_side_effect_query and ("lorundrostat" in source.lower() or "clinical" in source.lower()):
                continue
            filtered_docs.append(doc)
            if len(filtered_docs) >= 3:
                break
        
        # 만약 필터링 후 문서가 없다면 최소 1개는 유지 (안전장치)
        if not filtered_docs and docs_with_scores:
            filtered_docs = [docs_with_scores[0][0]]
        
        # 3. 문맥 생성 (Context)
        context = "\n\n".join([doc.page_content for doc in filtered_docs])
        
        # 4. 프롬프트 선택 및 구성 (Augment)
        prompt_template = get_prompt_template(mode)
        prompt = prompt_template.format(context=context, question=query)
        
        # 5. 생성 (Generate)
        raw_response_text = self.llm.invoke(prompt)
        
        # 6. 강제 번역 (Force Translation to Korean) - 영어나 한자 누수 방지
        try:
            translator = GoogleTranslator(source='auto', target='ko')
            translated_text = translator.translate(raw_response_text)
        except Exception as e:
            print(f"Translation Error: {e}")
            translated_text = raw_response_text # 번역 실패 시 원본 사용
            
        # 7. Prefix 결합
        prefix = "답변: 질문에 대한 한국어 설명은 다음과 같습니다.\n" if mode == "Doctor" else "답변: 환자분, 궁금하신 사항에 대한 한국어 설명은 다음과 같습니다.\n"
        final_answer = translated_text.strip()
        if not final_answer.startswith("답변:"):
            final_answer = prefix + final_answer
            
        return {
            "result": final_answer,
            "source_documents": filtered_docs
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
