import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import load_prompt

# API key불러오기
load_dotenv()

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")  # . 은 숨김폴더처리

# 파일 업로드 전용 폴더(임시저장폴더)
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("PDF 기반 QA")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화내용을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # 아무런 파일을 업로드 하지 않은 경우
    st.session_state["chain"] = None

# 사이드 바 생성
with st.sidebar:
    # 초기화버튼 생성
    clear_btn = st.button("대화내용 초기화")

    # 파일 업로드
    uploaded_file = st.file_uploader(
        "파일업로드", type=["pdf"], accept_multiple_files=True
    )

    # 모델 선택 메뉴
    selected_model = st.selectbox(
        "LLM 선택", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=0
    )

    # Temperature 슬라이더 추가
    temperature = st.slider(
        "응답의 다양성 조절 (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        key="temperature",
    )
    prompt = """사용자의 질문에 맞는 "정답"을 관련된 문서에서 찾아서 알려주세요. 정답 부분만 알려주면 됩니다. 답변은 PDF 문서 내에 무조건 있어야하고 관련된 문서가 없으면 답변을 하지 말고 모르겠다고 하세요"""
    user_text_prompt = st.text_area("프롬프트", value=prompt, key="user_text_prompt")

    user_text_apply_btn = st.button("적용", key="apply")


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메세지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
# @st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다..")


# cache_resource는 파일이 업로드 되면
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    all_docs = []
    for file in file:
        file_content = file.read()
        file_path = f"./.cache/files/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file_content)

        # 단계 1: 문서 로딩
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()

    return retriever


# 체인 생성
def create_chain(retriever, temperature, model_name="gpt-4o"):
    # prompt | llm | Outputparser
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = load_prompt("prompts/finaltest_rag.yaml", encoding="utf-8")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# 파일이 업로드 되었을 때
if uploaded_file:
    # 업로드 후 retriever 생성(작업시간 오래걸릴 예정)
    with st.spinner("문서 임베딩 처리 중... 잠시만 기다려주세요."):
        retriever = embed_file(uploaded_file)

# 적용버튼이 눌렸을 때
if user_text_apply_btn:
    loaded_prompt = load_prompt("prompts/finaltest_rag.yaml", encoding="utf8")
    prompt_template = user_text_prompt + loaded_prompt.template
    prompt = PromptTemplate.from_template(prompt_template)
    chain = create_chain(retriever, temperature=temperature, model_name=selected_model)
    st.session_state["chain"] = chain
    st.markdown(f"✅ 해당 프롬프트가 적용되었습니다")


# 초기화버튼이 눌리면..
if clear_btn:
    st.session_state["messages"] == []

# 이전 대화 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요.")

# 경고 메세지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 만약에 사용자 입력이 들어오면..
if user_input:
    # chain을 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)

        # 스트리밍 호출
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # 빈 공간을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""

            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화기록은 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)

    else:
        # 파일을 업로드 하라는 경고 메세지 출력
        warning_msg.error("파일을 업로드 해주세요.")
