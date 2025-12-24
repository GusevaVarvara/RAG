import os
import json
import time
import re
from glob import glob
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.prompts import PromptTemplate

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Set the GOOGLE_API_KEY environment variable")
SUBMISSION_NAME = "Guseva_v2"
EMAIL = "st119025@student.spbu.ru"
OUTPUT_FILENAME = f"submission_{SUBMISSION_NAME}.json"


def clean_answer(raw_answer):
    if not raw_answer:
        return "N/A"

    answer = str(raw_answer).strip()

    answer = re.sub(r'^(Answer|Output|Result):\s*', '', answer, flags=re.IGNORECASE).strip()
    answer = answer.split("\n")[0].strip()

    if answer.lower() in ["true", "yes", "correct"]: return "True"
    if answer.lower() in ["false", "no", "incorrect"]: return "False"
    if answer.upper() in ["N/A", "NO DATA", "NOT FOUND"]: return "N/A"

    answer = answer.replace('"', '').replace("'", '').replace("$", '').replace("€", '').replace("£", '')

    number_match = re.search(r"(-?[\d,]+(\.\d+)?)", answer)

    if number_match:
        clean_cand = number_match.group(1)
        clean_cand = clean_cand.replace(',', '')

        try:
            float(clean_cand)
            return clean_cand
        except ValueError:
            pass

    if len(answer.split()) > 5:
        return "N/A"

    return answer


def load_existing_progress():
    if not os.path.exists(OUTPUT_FILENAME):
        return []
    try:
        with open(OUTPUT_FILENAME, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("answers", [])
    except Exception as e:
        print(f"Ошибка чтения JSON: {e}")
        return []


def save_progress(answers):
    submission = {
        "team_email": EMAIL,
        "submission_name": SUBMISSION_NAME,
        "answers": answers
    }
    with open(OUTPUT_FILENAME, "w", encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=4)
    print(f"Прогресс сохранен ({len(answers)} ответов)")


def main():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists("faiss_index"):
        print("Загрузка индекса с диска.")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        print("Создание нового индекса.")
        pdf_files = glob("data/*.pdf")
        docs = []
        for pdf_path in pdf_files:
            file_name = os.path.basename(pdf_path)
            try:
                loader = PyMuPDFLoader(pdf_path)
                pages = loader.load()
                for page in pages:
                    page.metadata["pdf_sha1"] = file_name
                    page.metadata["page"] = page.metadata.get("page", 0) + 1
                docs.extend(pages)
            except Exception as e:
                print(f"Ошибка файла {pdf_path}: {e}")

        print(f"Всего загружено страниц: {len(docs)}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local("faiss_index")

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20}
    )

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.0,
    )

    template = """You are a specialized financial auditor. 
    Your goal is to extract precise values from the provided annual report context.

    CONTEXT:
    {context}

    QUESTION: 
    {question}

    INSTRUCTIONS:
    1. **Search for Tables:** Look for the requested data in financial tables first.
    2. **CHECK UNITS:** Check the top of the table/page for units (e.g., "(in thousands)", "(in millions)", "$000s").
       - If the table says "in thousands", multiply the number found by 1,000.
       - If the table says "in millions", multiply the number found by 1,000,000.
       - Example: If you see "Revenue: 204,191" and header says "in thousands", the answer is 204191000.
    3. **Date Check:** Ensure the data corresponds to the "end of the period" or the latest year mentioned in the report (usually 2021, 2022, or 2023).
    4. **Format:**
       - For numbers: Return ONLY the raw number (no currency symbols, no commas).
       - For Booleans: Return ONLY 'True' or 'False'.
       - If data is missing: Return 'N/A'.
    5. **Strict Constraint:** DO NOT return the year (e.g., "2022") as the value unless explicitly asked for a date.

    ANSWER:"""

    qa_chain_prompt = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_chain_prompt}
    )

    try:
        with open("questions.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            questions = data if isinstance(data, list) else data.get("questions", [])
    except:
        print("Не найден questions.json")
        return

    existing_answers = load_existing_progress()
    answers_list = existing_answers

    start_index = len(existing_answers)

    print(f"\nВсего вопросов: {len(questions)}")
    print(f"Уже отвечено: {start_index}")

    if start_index >= len(questions):
        return

    for i in range(start_index, len(questions)):
        q = questions[i]
        question_text = q['text'] if isinstance(q, dict) else str(q)

        print(f"\n[{i + 1}/{len(questions)}] Вопрос: {question_text}")

        try:
            time.sleep(15)

            result = qa_chain.invoke({"query": question_text})
            answer_text = clean_answer(result["result"])

            source_docs = result["source_documents"]

            if answer_text in ("N/A", "n/a", "False"):
                refs = []
            else:
                seen = set()
                refs = []
                for doc in source_docs:
                    key = (doc.metadata.get("pdf_sha1"), doc.metadata.get("page", 1))
                    if key in seen:
                        continue
                    seen.add(key)
                    refs.append({
                        "pdf_sha1": doc.metadata.get("pdf_sha1", ""),
                        "page_index": doc.metadata.get("page", 1)
                    })
                    if len(refs) >= 3:
                        break

            print(f"Ответ: {answer_text}")

            answers_list.append({
                "question_text": question_text,
                "value": answer_text,
                "references": refs
            })

            save_progress(answers_list)

        except Exception as e:
            error_message = str(e).lower()
            if "429" in error_message or "quota" in error_message or "resource_exhausted" in error_message:
                print("\nЛимит исчерпан.")
                print(f"Ошибка: {e}")
                break
            print(f"Ошибка: {e}")
            answers_list.append({"value": "error", "references": []})
            save_progress(answers_list)
            time.sleep(5)


if __name__ == "__main__":
    main()
