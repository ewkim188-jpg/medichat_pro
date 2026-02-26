
from model import qa_bot # model.py에서 챗봇 초기화 함수 가져오기
import sys
import time # 시간 측정을 위한 모듈

def verify():
    """
    챗봇이 정상적으로 작동하는지 확인하는 검증 함수
    """
    print("Initializing bot...") # 챗봇 초기화 시작 알림
    try:
        qa = qa_bot() # 챗봇 인스턴스 생성 (모델 로드 및 DB 연결)
        print("Bot initialized successfully!") # 초기화 성공 메시지
    except Exception as e: # 초기화 실패 시 에러 처리
        print(f"Error initializing bot: {e}")
        return # 에러 발생 시 함수 종료

    # 테스트할 질문 설정
    query = "What are the first-line treatments for hypertension?"
    print(f"\nQuerying: {query}") # 질문 출력
    
    start_time = time.time() # 시작 시간 기록
    try:
        # 질문을 던지고 답변 생성 (model.py의 qa_bot 반환 객체 사용)
        res = qa({'query': query}) 
        end_time = time.time() # 종료 시간 기록
        
        print("\n--- Response ---") # 답변 구분선
        print(res['result']) # 생성된 답변 출력
        
        # 소요 시간 출력 (성능 측정용)
        print(f"\nTime taken: {end_time - start_time:.2f} seconds")
        
        print("--- Sources ---") # 참조 문서 구분선
        # 답변 생성에 사용된 문서 출처들 출력
        for doc in res['source_documents']:
            print(f"- {doc.metadata.get('source', 'Unknown')}")
            
    except Exception as e: # 질의 과정 중 에러 발생 시 처리
        print(f"Error during query: {e}")

if __name__ == "__main__": # 스크립트 직접 실행 시 verify 함수 호출
    verify()
