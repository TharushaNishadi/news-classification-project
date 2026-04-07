import os
import pandas as pd
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import seaborn as sns
import re


CATEGORY_COLORS = {
    "Business": "#d4af37",          # gold
    "Opinion": "#3b82f6",           # blue
    "Political gossip": "#f97316",  # orange
    "Sports": "#22c55e",            # green
    "World news": "#a855f7"         # purple
}

# ---------------- ENV ----------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# ---------------- CSS ----------------
def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------- MODELS ----------------
@st.cache_data
def load_data(file):
    """Load CSV or Excel file and return DataFrame."""
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file type!")
            return None
        if df.empty:
            st.warning("Uploaded file is empty.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

@st.cache_resource
def load_classifier():
    """Load and return text classification model pipeline."""
    try:
        model = pipeline(
            "text-classification",
            model="Group-7/Assignment_1",
            device=-1
        )
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

@st.cache_resource
def load_qa_pipeline():
    return pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad",
        device=-1
    )

# ---------------- STATE ----------------
def init_state():
    st.session_state.setdefault("data", None)
    st.session_state.setdefault("qa_history", [])
    st.session_state.setdefault("question_input", "")
    st.session_state.setdefault("current_answer", "")
    st.session_state.setdefault("last_uploaded_file", None)

def clear_qa():
    st.session_state["qa_history"] = []
    st.session_state["current_answer"] = ""
    st.session_state["question_input"] = ""

def plot_cooccurrence(texts, max_features=50):
    # Convert text to term-document matrix
    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words='english'
    )
    
    X = vectorizer.fit_transform(texts)
    
    # Compute co-occurrence matrix
    Xc = (X.T * X)
    Xc.setdiag(0)  # remove self-cooccurrence
    
    # Convert to dense array
    cooc_matrix = Xc.toarray()
    
    # Get feature names
    words = vectorizer.get_feature_names_out()
    
    return cooc_matrix, words

def preprocess_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_top_keywords(df, category_col, text_col, top_n=10):
    results = {}

    for category in df[category_col].unique():
        texts = df[df[category_col] == category][text_col].fillna("").astype(str)

        if texts.empty:
            continue

        vectorizer = CountVectorizer(stop_words="english")
        X = vectorizer.fit_transform(texts)

        words = vectorizer.get_feature_names_out()
        counts = X.toarray().sum(axis=0)

        word_freq = dict(zip(words, counts))
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]

        results[category] = top_words

    return results

# ---------------- MAIN ----------------
def main():
    st.set_page_config(layout="wide")
    load_css()
    init_state()

    if "menu_open" not in st.session_state:
        st.session_state["menu_open"] = False

    # ---------------- HEADER ----------------
    header_col1, header_col2 = st.columns([1, 11])

    with header_col1:
        st.markdown('<div class="menu-wrapper">', unsafe_allow_html=True)
        if st.button("☰", key="menu_toggle_btn"):
            st.session_state["menu_open"] = not st.session_state["menu_open"]
        st.markdown('</div>', unsafe_allow_html=True)

    with header_col2:
        st.markdown("""
        <div class="header-banner">
            <div class="header-title">News Analysis Tool</div>
            <div class="header-sub">Classify news articles and perform question-answering</div>
        </div>
        """, unsafe_allow_html=True)

    #--------------Sidebar Style Info Panel-----------------#
    if st.session_state["menu_open"]:
        st.markdown('<div class="sidebar-panel">', unsafe_allow_html=True)

        st.markdown("### About")
        st.write("This application classifies news articles and answers questions based on selected article content.")

        st.markdown("### How to Use")
        st.write(
            "1. Upload a CSV or Excel file in the News Classification tab\n"
            "2. Review classification results\n"
            "3. Use the Q&A tab to ask questions"
        )

        st.markdown("### Data Requirements")
        st.write("- File type: CSV or Excel\n- Required column: content")

        st.markdown('</div>', unsafe_allow_html=True)

    # ================= CENTER PANEL (EXPANDED) =================
    center = st.container()
    with center:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        tabs = st.tabs(["📁 News Classification", "💬 Ask Questions", "📊 Analytics"])

        # TAB 1: CLASSIFICATION (FULL ORIGINAL LOGIC)
        with tabs[0]:
            # ================= FEATURED CATEGORIES =================
            st.markdown("""
            <div style="
                font-size:28px;
                font-weight:800;
                margin-bottom:15px;
                background: linear-gradient(135deg, #d4af37, #ffcc70, #b8860b);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;">
                Featured Categories
            </div>
            """, unsafe_allow_html=True)

            # --- CARD DATA ---
            categories_ui = [
                {
                    "name": "Business",
                    "desc": "Markets, companies, and financial updates",
                    "img": "https://images.unsplash.com/photo-1454165804606-c3d57bc86b40"
                    },
                    {
                        "name": "Opinion",
                        "desc": "Editorials and expert perspectives",
                        "img": "https://images.unsplash.com/photo-1504711434969-e33886168f5c"
                    },
                    {
                        "name": "Political gossip",
                        "desc": "Inside stories from politics",
                        "img": "https://images.unsplash.com/photo-1529107386315-e1a2ed48a620"
                    },
                    {
                        "name": "Sports",
                        "desc": "Matches, players, and highlights",
                        "img": "https://images.unsplash.com/photo-1517649763962-0c623066013b?crop=entropy&cs=tinysrgb&fit=crop&w=400&q=80"
                    },
                    {
                        "name": "World news",
                        "desc": "Global headlines and events",
                        "img": "https://images.unsplash.com/photo-1467269204594-9661b134dd2b"
                    }
            ]

            # --- GRID ---
            cols = st.columns(5)

            for i, cat in enumerate(categories_ui):
                with cols[i]:
                    st.markdown(f"""
                    <div class="category-card">
                        <img src="{cat['img']}" class="category-img"/>
                        <div class="category-title">{cat['name']}</div>
                        <div class="category-desc">{cat['desc']}</div>
                    
                    </div>
                    """, unsafe_allow_html=True)   
            st.subheader("Upload News File")
            st.write("Upload a CSV or Excel file that contains a column named `content`.")

            uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

            if uploaded_file is not None:
                df = load_data(uploaded_file)

                if df is None:
                    st.stop()

                # Reset Q&A state only when a new file is uploaded
                if st.session_state["last_uploaded_file"] != uploaded_file.name:
                    st.session_state["qa_history"] = []
                    st.session_state["current_answer"] = ""
                    st.session_state["question_input"] = ""
                    st.session_state["last_uploaded_file"] = uploaded_file.name

                # Normalize columns: strip whitespace + lowercase
                df.columns = [col.strip().lower() for col in df.columns]

                if "content" not in df.columns:
                    st.error("The uploaded file must contain a column named 'content' (case-insensitive).")
                    st.stop()

                df["content"] = df["content"].fillna("").astype(str)
                df["processed_content"] = df["content"].apply(preprocess_text)
                texts = df["processed_content"].tolist()

                # Load trained Hugging Face classification model
                # Each news article is passed through the model to predict its category
                with st.spinner("Classifying articles..."):
                    model = load_classifier()
                    if model is None:
                        st.stop()

                    try:
                        raw_preds = model(texts)
                        predictions = [p[0] if isinstance(p, list) else p for p in raw_preds]
                    except Exception as e:
                        st.error(f"Batch classification failed: {e}")
                        predictions = [{"label": "ERROR", "score": 0}] * len(texts)

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

                # Prepare downloadable output
                output_df = df[["content"]].copy()
                output_df["class"] = categories

                st.session_state["data"] = display_df

                # ------------------ Summary Metrics ------------------
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Articles", len(df))
                col2.metric("Categories", len(set(categories)))
                col3.metric("Avg Confidence", f"{round(sum(confidence)/len(confidence), 2)}%")

                # ------------------ Results ------------------
                st.subheader("Classification Results")
                st.dataframe(display_df, use_container_width=True)

                st.subheader("Category Distribution")
                st.table(pd.Series(categories).value_counts())

                # ------------------ Download Button ------------------
                st.download_button(
                    "Download Classified Output",
                    output_df.to_csv(index=False).encode("utf-8-sig"),
                    "output.csv",
                    "text/csv",
                    use_container_width=True
                )

                
        # TAB 2: Q&A
        with tabs[1]:

            if st.session_state["data"] is None or st.session_state["data"].empty:
                st.info("Please upload a CSV or Excel file.")
            else:

                df_display = st.session_state["data"]

                if "News Content" not in df_display.columns:
                    st.error("Column 'News Content' not found. Please upload and process data first.")
                    st.stop()

                articles = df_display["News Content"].fillna("").astype(str).tolist()
                article = st.selectbox("Select Article", articles)

                def run_suggested_question(question, article):
                    qa = load_qa_pipeline()

                    with st.spinner("Thinking..."):
                        res = qa({"question": question, "context": article})

                    st.session_state["question_input"] = question
                    st.session_state["current_answer"] = res["answer"]

                    new_item = {
                        "question": question,
                        "answer": res["answer"]
                    }

                    if not st.session_state["qa_history"] or st.session_state["qa_history"][-1] != new_item:
                        st.session_state["qa_history"].append(new_item)

                st.write("Suggested Questions")
                st.markdown('<div class="suggested-marker"></div>', unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)

                if c1.button("What is the main event?", key="q1"):
                    run_suggested_question("What is the main event?", article)

                if c2.button("Who is involved?", key="q2"):
                    run_suggested_question("Who is involved?", article)

                if c3.button("When did it happen?", key="q3"):
                    run_suggested_question("When did it happen?", article)

                if c4.button("Where did it happen?", key="q4"):
                    run_suggested_question("Where did it happen?", article)

                st.text_input("Enter question", key="question_input")

                colA, colB = st.columns(2)

                with colA:
                    ask = st.button("Get Answer", key="get_answer_btn")

                with colB:
                    st.button("Clear History", on_click=clear_qa, key="clear_history_btn")

                if ask:
                    q = st.session_state["question_input"].strip()

                    if q:
                        qa = load_qa_pipeline()
                        
                        with st.spinner("Thinking..."):
                            res = qa({"question": q, "context": article})

                        st.session_state["current_answer"] = res["answer"]

                        new_item = {
                            "question": q,
                            "answer": res["answer"]
                        }

                        if not st.session_state["qa_history"] or st.session_state["qa_history"][-1] != new_item:
                            st.session_state["qa_history"].append(new_item)
                    else:
                        st.warning("Please enter a question.")

                if st.session_state["current_answer"]:
                    st.success(st.session_state["current_answer"])

                if st.session_state["qa_history"]:
                    st.subheader("History")

                    for item in st.session_state["qa_history"]:
                        st.markdown(f"""
                        <div class="qa-history-box">
                            <div class="qa-question"><strong>Q:</strong> {item['question']}</div>
                            <div class="qa-answer"><strong>A:</strong> {item['answer']}</div>
                        </div>
                        """, unsafe_allow_html=True)


        # TAB 3: ANALYTICS 
        with tabs[2]:
            if st.session_state["data"] is None or st.session_state["data"].empty:
                st.info("Please upload a CSV or Excel file.")
            else:
                df_display = st.session_state["data"]
                # CHECK COLUMN EXISTS
                if "News Content" not in df_display.columns:
                    st.error("Column 'News Content' not found.")
                    st.stop()
                df = df_display
                st.subheader("Search & Filter")

                # Search input
                query = st.text_input("Search keyword")

                # Category filter (multi-select = better)
                selected_categories = st.multiselect(
                    "Select Category",
                    options=df["Category"].unique(),
                    default=df["Category"].unique()
                )

                # Start with full data
                filtered_df = df.copy()

                # Apply category filter
                filtered_df = filtered_df[filtered_df["Category"].isin(selected_categories)]

                # Apply search filter
                if query:
                    filtered_df = filtered_df[
                    filtered_df["News Content"].str.contains(query, case=False, na=False)
                ]
                    
                if filtered_df.empty:
                    st.warning("No records match the selected filters.")
                    st.stop()    

                # Show results
                st.subheader("Preview of your filtered data")
                st.dataframe(filtered_df, use_container_width=True)

                
                #-----------------Download-------------#
                st.subheader("Download Filtered Data")

                col_csv, col_json = st.columns(2)

                with col_csv:
                    st.download_button(
                        "📄 Download as CSV",
                        filtered_df.to_csv(index=False).encode("utf-8-sig"),
                        "filtered_data.csv",
                        use_container_width=True
                    )

                with col_json:
                    st.download_button(
                        "🧾 Download as JSON",
                        filtered_df.to_json(orient="records", indent=2).encode("utf-8"),
                        "filtered_data.json",
                        use_container_width=True
                )
                #-------------------Visuals----------------#
                col1, col2 = st.columns(2)
                # WordCloud
                with col1:
                    with st.container():
                        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        
                        st.markdown("#### Word Cloud")
        
                        text = " ".join(filtered_df["News Content"].fillna("").astype(str).tolist()).strip()

                        if text:
                            wc = WordCloud(width=500, height=500, background_color="black").generate(text)
                            fig, ax = plt.subplots(figsize=(5, 3))
                            ax.imshow(wc)
                            ax.axis("off")
                            st.pyplot(fig)
                        else:
                            st.info("Not enough text to generate word cloud.")

                        st.markdown('</div>', unsafe_allow_html=True)

                # HEATMAP CARD
                with col2:
                    with st.container():
                        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)

                        st.markdown("#### Heatmap")

                        texts = filtered_df["News Content"].fillna("").astype(str).tolist()

                        if len(" ".join(texts).split()) > 1:
                            cooc_matrix, words = plot_cooccurrence(texts)
                            cooc_df = pd.DataFrame(cooc_matrix, index=words, columns=words)

                            fig, ax = plt.subplots(figsize=(5, 3))
                            fig.patch.set_facecolor("black")
                            ax.set_facecolor("black")

                            sns.heatmap(cooc_df, cmap="coolwarm", square=True, ax=ax, cbar=False)
                            ax.tick_params(axis='x', colors='white', labelsize=6)
                            ax.tick_params(axis='y', colors='white', labelsize=6)
                            st.pyplot(fig)
                        else:
                            st.info("Not enough text to generate heatmap.")

                        st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("### Top Keywords by Category")

                top_keywords = get_top_keywords(filtered_df, "Category", "News Content", top_n=5)

                cols = st.columns(5)

                for i, (category, words) in enumerate(top_keywords.items()):
                    with cols[i]:
                        with st.container():
                            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)

                            st.markdown(f"#### {category}")

                            if words:
                                words_df = pd.DataFrame(words, columns=["Word", "Freq"])

                                fig, ax = plt.subplots(figsize=(3, 4))
                                fig.patch.set_facecolor("black")
                                ax.set_facecolor("black")
                                color = CATEGORY_COLORS.get(category, "#ffffff")

                                ax.bar(words_df["Word"], words_df["Freq"], color=color)

                                ax.tick_params(axis='x',colors='white', rotation=45, labelsize=6)
                                ax.tick_params(axis='y',colors='white', labelsize=6)

                                

                                st.pyplot(fig)

                            st.markdown('</div>', unsafe_allow_html=True)
                
                
                st.markdown("### Summary Charts")

                col1, col2 = st.columns(2)

                # PIE CHART
                with col1:
                    with st.container():
                        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)

                        st.markdown("#### Category Distribution")

                        counts = df["Category"].value_counts()
                        colors = [CATEGORY_COLORS.get(cat, "#ffffff") for cat in counts.index]
                        fig2, ax2 = plt.subplots(figsize=(4, 5))
                
                        wedges, _ = ax2.pie(counts, colors=colors, startangle=90)
                        fig2.patch.set_facecolor('black')
                        ax2.set_facecolor('black')

                        # Get total of counts
                        counts = df["Category"].value_counts()
                        total = counts.sum()

                        # Create legend labels with percentages
                        legend_labels = [f"{category} ({count/total*100:.1f}%)" for category, count in zip(counts.index, counts.values)]
                        colors = [CATEGORY_COLORS.get(cat, "#ffffff") for cat in counts.index]
                        # Create pie chart without percentages
                        wedges, texts = ax2.pie(
                            counts,
                            autopct=None,  # Remove percentages from pie
                            textprops={'color': 'white', 'fontsize': 9},
                            colors=colors,
                            startangle=90,
                            labels=None  # No labels on the pie
                        )

                        
                        # Add legend with percentages
                        ax2.legend(
                            wedges, 
                            legend_labels,  
                            title="Categories",
                            loc="center left",
                            bbox_to_anchor=(1, 0, 0.5, 1),
                            facecolor='black',
                            edgecolor='white',
                            labelcolor='white',
                            fontsize=16
                        )
                        ax2.get_legend().get_title().set_color('white')
                        st.markdown('</div>', unsafe_allow_html=True)


                        # Ensure pie is circular
                        ax2.axis('equal')

                        st.pyplot(fig2)
                #------------------- CONFIDENCE BAR-------------#
                with col2:
                    with st.container():
                        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)

                        st.markdown("#### Confidence by Category")

                        conf_df = df.groupby("Category")["Confidence (%)"].mean()
                        colors = [CATEGORY_COLORS.get(cat, "#ffffff") for cat in conf_df.index]

                        fig_conf, ax_conf = plt.subplots(figsize=(4, 3))
                        ax_conf.bar(conf_df.index, conf_df.values, color=colors)

                        ax_conf.set_facecolor("black")
                        fig_conf.patch.set_facecolor("black")

                        ax_conf.tick_params(axis='x', colors='white', rotation=30)
                        ax_conf.tick_params(axis='y', colors='white')

                        st.pyplot(fig_conf)
                        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()