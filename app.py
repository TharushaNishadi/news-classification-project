import os
import pandas as pd
import streamlit as st

# Configure environment for stable execution
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"


# ------------------ MODEL LOADING ------------------

@st.cache_resource
def load_classifier():
    """Load fine-tuned classification model."""
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model="Group-7/Assignment_1",
        device=-1
    )


@st.cache_resource
def load_qa_pipeline():
    """Load pre-trained question answering model."""
    from transformers import pipeline
    return pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad",
        device=-1
    )


# ------------------ DATA HANDLING ------------------

def load_data(file):
    """Read uploaded CSV or Excel file."""
    try:
        if file.name.endswith(".xlsx"):
            return pd.read_excel(file)
        if file.name.endswith(".csv"):
            file.seek(0)
            return pd.read_csv(file)

        st.error("Please upload a CSV or Excel file.")
        return None
    except Exception:
        st.error("There was an error reading the file.")
        return None


# ------------------ SESSION STATE ------------------

def init_state():
    """Initialize session variables."""
    if "data" not in st.session_state:
        st.session_state["data"] = None
    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""
    if "current_answer" not in st.session_state:
        st.session_state["current_answer"] = ""
    if "qa_history" not in st.session_state:
        st.session_state["qa_history"] = []


def set_question(q):
    """Set question from suggested buttons."""
    st.session_state["question_input"] = q


def clear_qa_history():
    """Clear question history."""
    st.session_state["qa_history"] = []
    st.session_state["current_answer"] = ""
    st.session_state["question_input"] = ""


# ------------------ MAIN APPLICATION ------------------

def main():
    st.set_page_config(page_title="News Analysis Tool", layout="wide")
    init_state()

    # Sidebar
    st.sidebar.title("About")
    st.sidebar.write(
        "This application classifies news articles and answers questions based on selected article content."
    )

    st.sidebar.subheader("How to Use")
    st.sidebar.write(
        "1. Upload a CSV or Excel file in the News Classification tab\n"
        "2. Review classification results\n"
        "3. Use the Q&A tab to ask questions"
    )

    st.sidebar.subheader("Data Requirements")
    st.sidebar.write(
        "- File type: CSV or Excel\n"
        "- Required column: content"
    )

    st.title("News Analysis Tool")
    st.caption("Classify news articles and perform question-answering.")

    tab1, tab2 = st.tabs(["News Classification", "Ask Questions"])

    # ------------------ TAB 1: CLASSIFICATION ------------------

    with tab1:
        st.subheader("Upload News File")

        # User instruction (important for usability)
        st.write("Upload a CSV or Excel file that contains a column named `content`.")

        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

        if uploaded_file is not None:
            df = load_data(uploaded_file)

            if df is None:
                return

            if "content" not in df.columns:
                st.error("The uploaded file must contain a column named 'content'.")
                return

            texts = df["content"].fillna("").astype(str).tolist()

            with st.spinner("Classifying articles..."):
                model = load_classifier()
                predictions = model(texts)

            # Map model labels to category names
            label_map = {
                "LABEL_0": "Business",
                "LABEL_1": "Opinion",
                "LABEL_2": "Political gossip",
                "LABEL_3": "Sports",
                "LABEL_4": "World news"
            }

            categories = [label_map.get(p["label"], p["label"]) for p in predictions]
            confidence = [round(p["score"] * 100, 2) for p in predictions]

            # Prepare display table
            display_df = df.copy()
            display_df.rename(columns={"content": "News Content"}, inplace=True)
            display_df["Category"] = categories
            display_df["Confidence (%)"] = confidence

            # Prepare output file
            output_df = df.copy()
            output_df["class"] = categories

            st.session_state["data"] = display_df

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Articles", len(df))
            col2.metric("Categories", len(set(categories)))
            col3.metric("Avg Confidence", f"{round(sum(confidence)/len(confidence),2)}%")

            st.subheader("Results")
            st.dataframe(display_df, use_container_width=True)

            st.subheader("Category Distribution")
            st.table(pd.Series(categories).value_counts())

            # Download classified output
            st.download_button(
                "Download Output",
                output_df.to_csv(index=False).encode("utf-8-sig"),
                "output.csv",
                "text/csv"
            )

    # ------------------ TAB 2: QUESTION ANSWERING ------------------

    with tab2:
        st.subheader("Ask Questions from News Articles")

        if st.session_state["data"] is None:
            st.info("Please upload and classify data first.")
            return

        df = st.session_state["data"]
        articles = df["News Content"].fillna("").astype(str).tolist()

        article = st.selectbox("Select Article", articles)

        st.write("Suggested Questions")

        # Suggested question buttons
        c1, c2, c3, c4 = st.columns(4)
        c1.button("What is the main event?", on_click=set_question, args=("What is the main event?",))
        c2.button("Who is involved?", on_click=set_question, args=("Who is involved?",))
        c3.button("When did it happen?", on_click=set_question, args=("When did it happen?",))
        c4.button("Where did it happen?", on_click=set_question, args=("Where did it happen?",))

        # Question input
        st.text_input("Enter your question", key="question_input")

        colA, colB = st.columns(2)

        with colA:
            get_answer = st.button("Get Answer", use_container_width=True)

        with colB:
            st.button("Clear History", use_container_width=True, on_click=clear_qa_history)

        # Generate answer
        if get_answer:
            question = st.session_state["question_input"].strip()

            if question == "":
                st.warning("Please enter a question.")
            else:
                with st.spinner("Generating answer..."):
                    qa = load_qa_pipeline()
                    result = qa({
                        "question": question,
                        "context": article
                    })
                    answer = result["answer"]

                st.session_state["current_answer"] = answer
                st.session_state["qa_history"].append({
                    "question": question,
                    "answer": answer
                })

        # Display answer
        if st.session_state["current_answer"]:
            st.subheader("Answer")
            st.markdown(
                f"<div style='background:#1f2937;padding:15px;border-radius:8px;'>{st.session_state['current_answer']}</div>",
                unsafe_allow_html=True
            )

        # Display question history
        if st.session_state["qa_history"]:
            st.subheader("Question History")

            for i, item in enumerate(st.session_state["qa_history"], start=1):
                st.markdown(
                    f"""
                    <div style="background:#111827;padding:15px;border-radius:8px;margin-bottom:10px;">
                        <strong>Question {i}:</strong> {item["question"]}<br>
                        <strong>Answer:</strong> {item["answer"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ------------------ RUN APP ------------------

if __name__ == "__main__":
    main()